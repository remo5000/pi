//! `exec`: Durable's `NodeExecutionEnv.exec` semantics (docs/semantics.md), output streamed as `output` events.

use crate::decode::StreamDecoder;
use crate::errors::Failure;
use crate::frame::{EVENT, Frame};
use crate::output::Output;
use crate::sys;
use crate::window::{Pending, Window};
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const EXIT_STDIO_GRACE: Duration = Duration::from_millis(100);
/// How often a delivery waiting for the link to drain looks again.
const UNSENT_POLL: Duration = Duration::from_millis(20);

/// Commands this daemon has not settled; the child may have been reaped while its pipes remain open.
pub type Groups = Arc<Mutex<HashMap<u32, Arc<AtomicBool>>>>;

pub enum Message {
    Data(usize, Vec<u8>),
    Eof(usize),
    Exit(ExitStatus),
    WaitFailed(String),
    /// Abort: the command settles as `aborted`.
    Cancel,
    /// Kill without aborting, as `cleanup()` does: the command settles with the killed process's status.
    Kill,
    /// The client finished its startup callback; this command's pipe readers may run.
    SpawnReady,
}

/// Cancellation of one request, registered before the request runs so a cancel that arrives first is not lost.
#[derive(Default)]
pub struct Control {
    pub aborted: AtomicBool,
    pub killed: AtomicBool,
    sender: Mutex<Option<Sender<Message>>>,
}

impl Control {
    pub fn cancel(&self, kill: bool) {
        let sender = self.sender.lock().unwrap();
        if kill {
            self.killed.store(true, Ordering::SeqCst);
        } else {
            self.aborted.store(true, Ordering::SeqCst);
        }
        if let Some(sender) = sender.as_ref() {
            let _ = sender.send(if kill { Message::Kill } else { Message::Cancel });
        }
    }

    fn attach(&self, sender: Sender<Message>) {
        *self.sender.lock().unwrap() = Some(sender);
    }

    pub fn spawn_ready(&self) {
        if let Some(sender) = self.sender.lock().unwrap().as_ref() {
            let _ = sender.send(Message::SpawnReady);
        }
    }
}

pub struct ExecRequest {
    pub command: Option<String>,
    pub argv: Option<Vec<String>>,
    pub cwd: String,
    pub env: Map<String, Value>,
    pub inherit_env: bool,
    pub shell_path: Option<String>,
    pub timeout: Option<Duration>,
    pub spill: Option<(u64, u64)>,
    pub spill_path: Option<String>,
    pub window: Option<Window>,
}

impl ExecRequest {
    pub fn from_json(json: &Value) -> Result<ExecRequest, Failure> {
        let strings = |value: &Value| -> Option<Vec<String>> {
            value
                .as_array()?
                .iter()
                .map(|item| item.as_str().map(str::to_string))
                .collect()
        };
        let spill_path = json["spill"]["path"].as_str().map(str::to_string);
        let spill = json["spill"].as_object().and_then(|spill| {
            Some((
                spill.get("afterBytes")?.as_u64()?,
                spill.get("afterLines")?.as_u64()?,
            ))
        });
        if spill_path.is_some() && spill.is_none() {
            return Err(Failure::new(
                "EINVAL",
                "exec needs valid spill thresholds for an explicit log",
            ));
        }
        Ok(ExecRequest {
            command: json["command"].as_str().map(str::to_string),
            argv: strings(&json["argv"]),
            cwd: json["cwd"]
                .as_str()
                .ok_or_else(|| Failure::new("EINVAL", "exec needs cwd"))?
                .to_string(),
            env: json["env"].as_object().cloned().unwrap_or_default(),
            inherit_env: json["inheritEnv"].as_bool().unwrap_or(true),
            shell_path: json["shellPath"].as_str().map(str::to_string),
            timeout: json["timeoutMs"]
                .as_f64()
                .map(|ms| Duration::from_secs_f64(ms / 1000.0)),
            spill,
            spill_path,
            window: Window::from_json(&json["window"]),
        })
    }
}

pub fn kill_group(pid: u32, reaped: bool) {
    // On Windows taskkill names the leader by PID and cannot safely target its descendants once it was reaped.
    if reaped && cfg!(windows) {
        return;
    }
    sys::kill_tree(pid);
}

fn random_uuid() -> String {
    let mut bytes = sys::random_bytes();
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// Spill of the complete raw output, started once it crosses either threshold, like Node's.
struct Spill {
    after_bytes: u64,
    after_lines: u64,
    seen_bytes: u64,
    seen_newlines: u64,
    prefix: Vec<Vec<u8>>,
    file: Option<File>,
    path: Option<String>,
    failed: Option<String>,
}

impl Spill {
    fn push(&mut self, chunk: &[u8], tmpdir: &str) {
        if self.failed.is_some() || chunk.is_empty() {
            return;
        }
        if self.file.is_none() {
            self.seen_bytes += chunk.len() as u64;
            self.seen_newlines += chunk.iter().filter(|byte| **byte == b'\n').count() as u64;
            let lines = self.seen_newlines + u64::from(chunk.last() != Some(&b'\n'));
            if self.seen_bytes <= self.after_bytes && lines <= self.after_lines {
                self.prefix.push(chunk.to_vec());
                return;
            }
            if let Err(error) = self.start(tmpdir) {
                self.failed = Some(error);
                return;
            }
        }
        if let Some(file) = &mut self.file
            && let Err(error) = file.write_all(chunk)
        {
            self.failed = Some(error.to_string());
        }
    }

    /// `createTempFile({ prefix: "pi-output-", suffix: ".log" })`: a fresh `tmp-` directory holding the file.
    fn start(&mut self, tmpdir: &str) -> Result<(), String> {
        let prefix = Path::new(tmpdir)
            .join("tmp-")
            .to_string_lossy()
            .into_owned();
        let directory = sys::mkdtemp(&prefix).map_err(|error| error.to_string())?;
        let path = Path::new(&directory)
            .join(format!("pi-output-{}.log", random_uuid()))
            .to_string_lossy()
            .into_owned();
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&path)
            .map_err(|error| error.to_string())?;
        for chunk in self.prefix.drain(..) {
            file.write_all(&chunk).map_err(|error| error.to_string())?;
        }
        self.file = Some(file);
        self.path = Some(path);
        Ok(())
    }
}

fn spawn(
    request: &ExecRequest,
    program: &str,
    args: &[String],
    piped_stdin: bool,
) -> std::io::Result<Child> {
    let mut command = Command::new(program);
    command
        .current_dir(&request.cwd)
        .stdin(if piped_stdin {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    sys::configure(&mut command, args, &request.env, request.inherit_env);
    let child = command.spawn()?;
    sys::after_spawn(&child);
    Ok(child)
}

/// Read one pipe on its own thread. Without an output window, reading waits while too much output is unsent.
fn spawn_reader(
    stream: usize,
    pipe: impl sys::Pipe + Send + 'static,
    sender: Sender<Message>,
    stop: Arc<AtomicBool>,
    output: Option<Arc<Output>>,
    startup: Arc<(Mutex<bool>, Condvar)>,
    paused_readers: Arc<AtomicUsize>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        // The OS pipe bounds output while the client persists startup; other requests keep flowing.
        let (lock, ready) = &*startup;
        let mut acknowledged = lock.lock().unwrap();
        while !*acknowledged {
            acknowledged = ready.wait(acknowledged).unwrap();
        }
        drop(acknowledged);
        sys::read_pipe(pipe, &stop, |bytes| {
            let delivered = sender.send(Message::Data(stream, bytes.to_vec())).is_ok();
            if let Some(output) = &output {
                paused_readers.fetch_add(1, Ordering::SeqCst);
                output.wait_for_room(&stop);
                paused_readers.fetch_sub(1, Ordering::SeqCst);
            }
            delivered
        });
        let _ = sender.send(Message::Eof(stream));
    })
}

fn output_frame(id: u32, stream: &str, text: String, skipped: Option<Value>) -> Frame {
    let mut json = json!({ "kind": "output", "stream": stream });
    if let Some(skipped) = skipped {
        json["skipped"] = skipped;
    }
    Frame::with_payload(EVENT, id, json, text.into_bytes())
}

/// Run one command, sending output events for request `id`, until it settles.
pub fn run(
    id: u32,
    request: ExecRequest,
    tmpdir: &str,
    output: &Arc<Output>,
    control: &Control,
    groups: &Groups,
) -> Result<Value, Failure> {
    let (sender, receiver) = mpsc::channel();
    control.attach(sender.clone());
    if control.aborted.load(Ordering::SeqCst) {
        return Err(Failure::new("aborted", "aborted"));
    }
    // A string runs through the shell (as its last argument, or on stdin for legacy WSL bash); argv runs directly.
    let (program, args, stdin_command) = match (&request.command, &request.argv) {
        (Some(command), _) => {
            let shell = sys::shell_config(request.shell_path.as_deref())?;
            let mut args = shell.args;
            if shell.command_on_stdin {
                (shell.program, args, Some(command.clone()))
            } else {
                args.push(command.clone());
                (shell.program, args, None)
            }
        }
        (None, Some(argv)) => match argv.split_first() {
            Some((first, rest)) => (
                sys::resolve_program(first, &request.cwd, &request.env)?,
                rest.to_vec(),
                None,
            ),
            None => return Err(Failure::new("spawn_error", "Empty argv: no program to run")),
        },
        (None, None) => return Err(Failure::new("EINVAL", "exec needs command or argv")),
    };
    if !Path::new(&request.cwd).exists() {
        return Err(Failure::new(
            "spawn_error",
            format!(
                "Working directory does not exist: {}\nCannot execute bash commands.",
                request.cwd
            ),
        ));
    }
    // An explicit log is opened without create/truncate before spawning, including for an empty command.
    let explicit_log = request
        .spill_path
        .as_ref()
        .map(|path| {
            OpenOptions::new().append(true).open(path).map_err(|error| {
                Failure::new(
                    "unknown",
                    format!("Failed to preserve complete shell output: {error}"),
                )
            })
        })
        .transpose()?;
    let mut child = spawn(&request, &program, &args, stdin_command.is_some()).map_err(|error| {
        let failure = Failure::new("spawn_error", error.to_string());
        match &request.spill_path {
            Some(path) => failure.extra(json!({ "spillPath": path })),
            None => failure,
        }
    })?;
    let pid = child.id();
    let reaped = Arc::new(AtomicBool::new(false));
    groups.lock().unwrap().insert(pid, reaped.clone());
    // This frame names the command child, never this daemon or the SSH transport.
    output.bulk(
        Frame::new(EVENT, id, json!({ "kind": "spawned", "pid": pid })),
        None,
    );
    if let (Some(command), Some(mut stdin)) = (stdin_command, child.stdin.take()) {
        let _ = stdin.write_all(command.as_bytes());
    }
    let deadline = request.timeout.map(|timeout| Instant::now() + timeout);
    let stop = Arc::new(AtomicBool::new(false));
    let startup = Arc::new((Mutex::new(false), Condvar::new()));
    let paused_readers = Arc::new(AtomicUsize::new(0));
    let release_readers = || {
        let (lock, ready) = &*startup;
        *lock.lock().unwrap() = true;
        ready.notify_all();
    };
    // Without a window, output is not coalesced; reading then waits for the link.
    let gate = if request.window.is_none() {
        Some(output.clone())
    } else {
        None
    };
    let readers = [
        spawn_reader(
            0,
            child.stdout.take().unwrap(),
            sender.clone(),
            stop.clone(),
            gate.clone(),
            startup.clone(),
            paused_readers.clone(),
        ),
        spawn_reader(
            1,
            child.stderr.take().unwrap(),
            sender.clone(),
            stop.clone(),
            gate,
            startup.clone(),
            paused_readers.clone(),
        ),
    ];
    let waiter = sender.clone();
    let wait_reaped = reaped.clone();
    thread::spawn(move || {
        let status = child.wait();
        // The Child handle is still held here, so Windows cannot reuse its PID before this flag is visible.
        wait_reaped.store(true, Ordering::SeqCst);
        let _ = waiter.send(match status {
            Ok(status) => Message::Exit(status),
            Err(error) => Message::WaitFailed(error.to_string()),
        });
    });
    drop(sender);

    let mut decoders = [StreamDecoder::new(), StreamDecoder::new()];
    let mut spill = request.spill.map(|(after_bytes, after_lines)| Spill {
        after_bytes,
        after_lines,
        seen_bytes: 0,
        seen_newlines: 0,
        prefix: Vec::new(),
        file: explicit_log,
        path: request.spill_path,
        failed: None,
    });
    let mut pending = request.window.map(Pending::new);
    let unsent = Arc::new(AtomicUsize::new(0));
    let names = ["stdout", "stderr"];
    let emit = |pending: &mut Option<Pending>, stream: usize, text: String| {
        if text.is_empty() {
            return;
        }
        match pending {
            Some(pending) => pending.push(names[stream], text),
            None => output.bulk(output_frame(id, names[stream], text, None), None),
        }
    };
    let deliver = |pending: &mut Pending| {
        for event in pending.take() {
            output.bulk(
                output_frame(id, event.stream, event.text, event.skipped),
                Some(unsent.clone()),
            );
        }
    };
    let mut ended = [false, false];
    let mut status: Option<Result<ExitStatus, String>> = None;
    let mut idle_until: Option<Instant> = None;
    let mut timed_out = false;
    let mut aborted = false;
    let mut killed = false;
    let kill = |killed: &mut bool| {
        if !*killed {
            *killed = true;
            kill_group(pid, reaped.load(Ordering::SeqCst));
        }
    };
    if control.killed.load(Ordering::SeqCst) {
        kill(&mut killed);
    }
    loop {
        let now = Instant::now();
        // Node's timer stays armed until the command settles, including the grace period after exit.
        if !timed_out && deadline.is_some_and(|deadline| now >= deadline) {
            timed_out = true;
            kill(&mut killed);
        }
        if status.is_some() && ended[0] && ended[1] {
            break;
        }
        let startup_ready = *startup.0.lock().unwrap();
        if startup_ready && idle_until.is_some_and(|idle| now >= idle) {
            if paused_readers.load(Ordering::SeqCst) > 0 {
                // A full link paused this command's reader; its pipe can still contain unlogged bytes.
                idle_until = Some(now + EXIT_STDIO_GRACE);
            } else {
                // A descendant may hold the pipes open after the process exited; stop waiting for it.
                break;
            }
        }
        if let Some(pending) = &mut pending
            && !pending.is_empty()
            && now >= pending.next_send()
            && unsent.load(Ordering::SeqCst) == 0
        {
            deliver(pending);
        }
        let mut wait = Duration::from_secs(3600);
        if let Some(deadline) = deadline.filter(|_| !timed_out) {
            wait = wait.min(deadline.saturating_duration_since(now));
        }
        if startup_ready && let Some(idle) = idle_until {
            wait = wait.min(idle.saturating_duration_since(now));
        }
        if let Some(pending) = pending.as_ref().filter(|pending| !pending.is_empty()) {
            wait = wait.min(if unsent.load(Ordering::SeqCst) > 0 {
                UNSENT_POLL
            } else {
                pending.next_send().saturating_duration_since(now)
            });
        }
        match receiver.recv_timeout(wait) {
            Ok(Message::Data(stream, bytes)) => {
                emit(&mut pending, stream, decoders[stream].decode(&bytes, false));
                if let Some(spill) = &mut spill {
                    spill.push(&bytes, tmpdir);
                    if spill.failed.is_some() {
                        kill(&mut killed);
                    }
                }
                if status.is_some() {
                    idle_until = Some(Instant::now() + EXIT_STDIO_GRACE);
                }
            }
            Ok(Message::Eof(stream)) => ended[stream] = true,
            Ok(Message::Exit(exit)) => {
                status = Some(Ok(exit));
                idle_until = Some(Instant::now() + EXIT_STDIO_GRACE);
            }
            Ok(Message::WaitFailed(error)) => {
                status = Some(Err(error));
                idle_until = Some(Instant::now());
            }
            Ok(Message::Cancel) => {
                aborted = true;
                kill(&mut killed);
            }
            Ok(Message::Kill) => kill(&mut killed),
            Ok(Message::SpawnReady) => {
                release_readers();
                if status.is_some() {
                    idle_until = Some(Instant::now() + EXIT_STDIO_GRACE);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    // Like Node destroying its streams: stop reading, so the pipes close even if a descendant still holds them.
    stop.store(true, Ordering::SeqCst);
    release_readers();
    for reader in &readers {
        sys::interrupt_reader(reader);
    }
    groups.lock().unwrap().remove(&pid);
    emit(&mut pending, 0, decoders[0].decode(&[], true));
    emit(&mut pending, 1, decoders[1].decode(&[], true));
    if let Some(pending) = &mut pending
        && !pending.is_empty()
    {
        deliver(pending);
    }
    let spill_path = spill.as_ref().and_then(|spill| spill.path.clone());
    let with_spill = |failure: Failure| match &spill_path {
        Some(path) => failure.extra(json!({ "spillPath": path })),
        None => failure,
    };
    if timed_out {
        return Err(with_spill(Failure::new("timeout", "timeout")));
    }
    if aborted {
        return Err(with_spill(Failure::new("aborted", "aborted")));
    }
    if let Some(error) = spill.as_ref().and_then(|spill| spill.failed.clone()) {
        return Err(with_spill(Failure::new(
            "unknown",
            format!("Failed to preserve complete shell output: {error}"),
        )));
    }
    let status = match status {
        Some(Ok(status)) => status,
        Some(Err(error)) => return Err(with_spill(Failure::new("spawn_error", error))),
        None => {
            return Err(with_spill(Failure::new(
                "spawn_error",
                "process did not report its exit",
            )));
        }
    };
    let mut result = json!({ "exitCode": sys::exit_code(status) });
    if let Some(path) = spill_path {
        result["spillPath"] = json!(path);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_explicit_log_in_backend_namespace() {
        let request = ExecRequest::from_json(&json!({
            "cwd": "/work",
            "argv": ["sh", "-c", "exit 0"],
            "spill": { "path": "/work/caller-created.log", "afterBytes": 500, "afterLines": 10 }
        }))
        .unwrap();
        assert_eq!(request.spill, Some((500, 10)));
        assert_eq!(
            request.spill_path.as_deref(),
            Some("/work/caller-created.log")
        );
    }

    #[test]
    fn refuses_explicit_log_without_valid_thresholds() {
        for spill in [
            json!({ "path": "/work/log" }),
            json!({ "path": "/work/log", "afterBytes": -1, "afterLines": 10 }),
        ] {
            let error =
                ExecRequest::from_json(&json!({ "cwd": "/work", "argv": ["sh"], "spill": spill }))
                    .err()
                    .unwrap();
            assert_eq!(error.code, "EINVAL");
        }
    }

    #[cfg(unix)]
    #[test]
    fn drains_eager_log_when_child_exits_while_reader_waits_for_link() {
        let directory = std::env::temp_dir().join(format!("pi-env-backpressure-{}", random_uuid()));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("complete.log");
        std::fs::write(&path, []).unwrap();
        let request = ExecRequest::from_json(&json!({
            "cwd": directory,
            "argv": ["sh", "-c", "printf first; sleep 0.15; printf FINISH"],
            "spill": { "path": path, "afterBytes": 1000, "afterLines": 1000 }
        }))
        .unwrap();
        let output = Arc::new(Output::default());
        // Fill the slow transport before this command starts, so its first read blocks before the last bytes arrive.
        output.bulk(
            Frame::with_payload(
                EVENT,
                99,
                json!({ "kind": "output", "stream": "stdout" }),
                vec![b'z'; 4 * 1024 * 1024 + 1],
            ),
            None,
        );
        let control = Arc::new(Control::default());
        let groups = Arc::new(Mutex::new(HashMap::new()));
        let (done, finished) = mpsc::channel();
        let worker = {
            let output = output.clone();
            let control = control.clone();
            let groups = groups.clone();
            thread::spawn(move || {
                let result = run(7, request, &sys::tmpdir(), &output, &control, &groups);
                done.send(result).unwrap();
            })
        };
        while control.sender.lock().unwrap().is_none() {
            thread::sleep(Duration::from_millis(1));
        }
        control.spawn_ready();
        // No writer drains the link until after the parent exits and its 100 ms stdio grace elapses.
        thread::sleep(Duration::from_millis(450));
        let writer = {
            let output = output.clone();
            thread::spawn(move || {
                let mut frames = Vec::new();
                output.run(&mut frames);
            })
        };
        let result = finished
            .recv_timeout(Duration::from_secs(10))
            .unwrap()
            .unwrap();
        output.close();
        worker.join().unwrap();
        writer.join().unwrap();
        let content = std::fs::read(&path).unwrap();
        std::fs::remove_dir_all(directory).unwrap();
        assert_eq!(result["exitCode"], 0);
        assert_eq!(content, b"firstFINISH");
    }
}
