import type { Context, JsonValue } from "@earendil-works/chord";
import { awaitWithContext, withAbortSignal } from "@earendil-works/chord/context";
import { defineDoc } from "../documents.ts";
import type { CommandResult, CommandState } from "../env/command-state.ts";
import { boundOutput, OutputBuffer, PROGRESS_BYTES_PER_SECOND, Progress, sanitizeOutput } from "../harness/output.ts";
import type { ToolTaskResult } from "../harness/tool.ts";
import { createToolExecutionApi } from "../harness/tool-api.ts";
import type { ToolDiagnostic, ToolExecutionApi, ToolExecutionResult, ToolRegistration } from "../harness/types.ts";
import { defineTask } from "../tasks.ts";
import { DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, utf8ByteLength } from "../truncate.ts";
import type { JsonObject, TaskId, TaskRuntime } from "../types.ts";

export type ManagedCommand = { cwd: string; fullOutputPath: string; onSpawn(pid?: number): Promise<void> };
export const managedCommand = Symbol("managed Bash invocation");
export type ManagedCommandApi = ToolExecutionApi & { [managedCommand]?: ManagedCommand };
// Spread-based wrappers retain the factory identity; the job invokes the selected wrapper exactly once.
export const commandFactory = Symbol("bash command factory");
export type CommandTool = ToolRegistration & { [commandFactory]?: string };

type CommandInput = {
	command: string;
	timeout?: number;
	caller: TaskId<ToolTaskResult>;
	callId: string;
	toolName: string;
	cwd: string;
	environmentId: string;
	fullOutputPath: string;
	factoryId: string;
	yieldTimeMs: number;
};
type CommandCheckpoint = { phase: "start" } | { phase: "execute" } | { phase: "report"; state: CommandState };
type Runtime = TaskRuntime<CommandInput, CommandCheckpoint, CommandState, object>;

export const CommandDoc = defineDoc<{
	state?: CommandState;
	yielded: boolean;
	details?: JsonValue;
	pid?: number;
	diagnostics?: ToolDiagnostic[];
}>({
	kind: "pi.command",
	version: 1,
	scope: "task",
	initial: () => ({ yielded: false }),
});

const limits = { maxBytes: DEFAULT_MAX_BYTES, maxLines: DEFAULT_MAX_LINES, retain: "tail" } as const;

/** One background execution, with committed progress and a replay-safe completion report, not a second task registry. */
export const BashCommandTask = defineTask<CommandInput, CommandCheckpoint, CommandState>({
	name: "pi.bash-command",
	version: 1,
	initial: () => ({ phase: "start" }),
	phases: {
		start: async (task, runtime, context) => {
			// Recovery at execute must never repeat a command whose external effects may already have happened.
			await runtime.commit(() => ({ status: "running", checkpoint: { phase: "execute" } }), context);
			await executeCommand(task.input, runtime, context);
		},
		execute: async (task, runtime, context) => {
			const previous = (await runtime.snapshot(CommandDoc, task.id, context))?.state;
			const state: CommandState = {
				output: "",
				truncated: previous?.truncated ?? false,
				wall_time_seconds: previous?.wall_time_seconds ?? 0,
				full_output_path: task.input.fullOutputPath,
				...(previous?.status === "running" && previous.pid !== undefined ? { pid: previous.pid } : {}),
				status: "failed",
				error: "Command was interrupted and may have partially run; it was not restarted",
			};
			await finishExecution(runtime, state, task.input.caller, context);
		},
		report: async (task, runtime, context) => {
			const state = task.state.checkpoint.state;
			const doc = await runtime.snapshot(CommandDoc, task.id, context);
			if (
				doc?.yielded &&
				(await callerReturnedRunning(task.input.caller, task.input.fullOutputPath, runtime, context))
			) {
				const conversation = await runtime.conversation(runtime.conversationId, context);
				if (conversation === undefined) throw new Error("Command conversation does not exist");
				await conversation.submit(
					{
						type: "input",
						content: formatCommand(state, task.input.command, true, doc.pid),
						whenBusy: "followUp",
						requestId: `bash-completion:${task.id}`,
					},
					context,
				);
			}
			await runtime.commit(() => ({ status: "terminal", outcome: { status: "completed", result: state } }), context);
		},
	},
	abort: async (_task, runtime, context) => {
		await runtime.commit(() => ({ status: "terminal", outcome: { status: "aborted" } }), context);
	},
});

async function callerReturnedRunning(
	id: TaskId<ToolTaskResult>,
	fullOutputPath: string,
	runtime: Runtime,
	context: Context,
): Promise<boolean> {
	const caller = await runtime.waitForTask(id, context);
	if (caller.state.outcome.status !== "completed") return false;
	const entry = await runtime.entry(caller.state.outcome.result.entryId, context);
	const message = entry?.model?.[0];
	if (message?.role !== "toolResult" || message.isError) return false;
	const details = message.details;
	return (
		typeof details === "object" &&
		details !== null &&
		"status" in details &&
		details.status === "running" &&
		"full_output_path" in details &&
		details.full_output_path === fullOutputPath
	);
}

async function executeCommand(input: CommandInput, runtime: Runtime, context: Context): Promise<void> {
	const controller = new AbortController();
	const signal = AbortSignal.any([runtime.signal, controller.signal]);
	const executionContext = withAbortSignal(signal, context);
	const output = new OutputBuffer(limits);
	let ended = false;
	const assertLive = () => {
		if (ended) throw new Error("Background Bash invocation has ended");
	};
	const diagnostics: ToolDiagnostic[] = [];
	let pid: number | undefined;
	let running = false;
	const startedAt = Date.now();
	const snapshot = (): CommandState => {
		const retained = output.snapshot();
		return {
			status: running ? "running" : "starting",
			...(pid === undefined ? {} : { pid }),
			output: retained.text,
			truncated: retained.droppedBytes > 0,
			wall_time_seconds: (Date.now() - startedAt) / 1000,
			full_output_path: input.fullOutputPath,
		};
	};
	const progress = new Progress(
		async () => {
			const state = snapshot();
			await runtime.commit(async (tx) => {
				const doc = await tx.doc(CommandDoc, runtime.taskId);
				doc.state = state;
				if (pid !== undefined) doc.pid = pid;
				if (diagnostics.length > 0) doc.diagnostics = diagnostics;
				return undefined;
			}, context);
			return utf8ByteLength(state.output);
		},
		(error) => {
			if (!runtime.signal.aborted) runtime.report(error);
		},
		runtime.settings.progress.outputIntervalMs,
	);

	const supervisionFailed = (error: unknown) => {
		controller.abort(error);
		if (!runtime.signal.aborted) runtime.report(error);
	};
	let callerDone = false;
	const caller = callerReturnedRunning(input.caller, input.fullOutputPath, runtime, context).then((allowed) => {
		callerDone = true;
		if (!allowed) controller.abort(new Error("Originating Bash call did not yield successfully"));
		return allowed;
	});
	caller.catch((error: unknown) => {
		callerDone = true;
		supervisionFailed(error);
	});
	const monitor = (async () => {
		while (!callerDone) {
			const original = await runtime.getTask(input.caller, context);
			if (original === undefined || original.abortRequested)
				controller.abort(new Error("Originating Bash call aborted"));
			await runtime.sleep(runtime.now() + 20, context);
		}
	})();
	monitor.catch(supervisionFailed);

	let state: CommandState;
	try {
		const env = await runtime.env(executionContext);
		if (env === undefined) throw new Error("No execution environment is configured");
		if (env.id !== input.environmentId) throw new Error("Execution environment changed before command startup");
		const tool = (await runtime.agent(executionContext)).tools.find((tool) => tool.name === input.toolName) as
			| CommandTool
			| undefined;
		if (tool?.[commandFactory] !== input.factoryId)
			throw new Error("The Bash implementation changed before command startup");
		const api = createToolExecutionApi(
			runtime,
			{
				callId: input.callId,
				output: (chunk, skipped) => {
					assertLive();
					if (output.push(chunk, skipped)) progress.mark();
				},
				outputWindow: {
					...limits,
					minIntervalMs: runtime.settings.progress.outputIntervalMs,
					bytesPerSecond: PROGRESS_BYTES_PER_SECOND,
				},
				diagnostic: (diagnostic) => {
					assertLive();
					diagnostics.push(diagnostic);
				},
				details: async (value, detailsContext) => {
					assertLive();
					await runtime.commit(async (tx) => {
						(await tx.doc(CommandDoc, runtime.taskId)).details = value;
						return undefined;
					}, detailsContext);
				},
			},
			env,
		);
		Object.assign(api, {
			[managedCommand]: {
				cwd: input.cwd,
				fullOutputPath: input.fullOutputPath,
				onSpawn: async (spawnedPid: number | undefined) => {
					assertLive();
					if (running) throw new Error("Execution backend reported startup more than once");
					if (spawnedPid !== undefined && (!Number.isSafeInteger(spawnedPid) || spawnedPid <= 0)) {
						throw new Error("Execution backend reported an invalid PID");
					}
					pid = spawnedPid;
					running = true;
					await progress.markAndWait();
				},
			},
		});
		const result = await tool.execute(
			{
				command: input.command,
				yield_time_ms: input.yieldTimeMs,
				...(input.timeout === undefined ? {} : { timeout: input.timeout }),
			},
			api,
			executionContext,
		);
		const details = result.details;
		if (
			typeof details !== "object" ||
			details === null ||
			!("status" in details) ||
			details.status !== "exited" ||
			!("exit_code" in details) ||
			typeof details.exit_code !== "number" ||
			!Number.isSafeInteger(details.exit_code)
		) {
			throw new Error("The Bash implementation did not return an exited command result");
		}
		if (result.isError && details.exit_code === 0) throw new Error("The background Bash invocation failed");
		const retained =
			result.content === undefined
				? output.snapshot()
				: boundOutput(
						sanitizeOutput(result.content.flatMap((item) => (item.type === "text" ? [item.text] : [])).join("")),
						limits,
					);
		state = {
			status: "exited",
			exit_code: details.exit_code,
			output: retained.text,
			truncated: retained.droppedBytes > 0,
			full_output_path: input.fullOutputPath,
			wall_time_seconds: (Date.now() - startedAt) / 1000,
		};
	} catch (error) {
		if (runtime.signal.aborted) throw error;
		state = {
			...snapshot(),
			output: "",
			status: "failed",
			error: error instanceof Error ? error.message : String(error),
		};
	} finally {
		ended = true;
		output.end();
		for (const waiter of await progress.stop()) waiter.reject(new Error("Command progress ended"));
		if (runtime.signal.aborted) {
			controller.abort(runtime.signal.reason);
			await Promise.allSettled([caller, monitor]);
		}
	}
	await finishExecution(runtime, state, input.caller, context);
	await Promise.allSettled([monitor, caller]);
}

async function finishExecution(
	runtime: Runtime,
	state: CommandState,
	caller: TaskId<ToolTaskResult>,
	context: Context,
): Promise<void> {
	await runtime.commit(async (tx) => {
		(await tx.doc(CommandDoc, runtime.taskId)).state = state;
		return { status: "running", checkpoint: { phase: "report", state } };
	}, context);
	// Keep the task document alive until the original result commits, including fast commands that did not yield.
	await runtime.waitForTask(caller, context);
}

export async function yieldBashCommand(
	args: { command: string; timeout?: number; yield_time_ms: number },
	api: ToolExecutionApi,
	context: Context,
	tool: CommandTool,
): Promise<ToolExecutionResult<CommandResult>> {
	const startedAt = Date.now();
	const env = api.env;
	if (env === undefined) throw new Error("No execution environment is configured");
	if (env.reportsSpawn !== true)
		throw new Error("This execution backend does not report startup; yielding is unsupported");
	const factoryId = tool[commandFactory];
	if (factoryId === undefined) throw new Error("The Bash implementation does not support background execution");
	const created = await env.createTempFile({ prefix: "pi-bash-", suffix: ".log" }, context);
	if (!created.ok) throw created.error;
	const fullOutputPath = created.value;
	// Reserve command, error, and status metadata before admitting a launch that must report a bounded result.
	if (utf8ByteLength(fullOutputPath) > DEFAULT_MAX_BYTES - 4 * 1024)
		throw new Error("Execution backend output path is too long for a bounded Bash result");
	const id = await api.commit(async (tx) => {
		const id = await tx.createTask(
			BashCommandTask,
			{
				command: args.command,
				...(args.timeout === undefined ? {} : { timeout: args.timeout }),
				caller: api.taskId as TaskId<ToolTaskResult>,
				callId: api.callId,
				toolName: tool.name,
				factoryId,
				yieldTimeMs: args.yield_time_ms,
				cwd: env.cwd,
				environmentId: env.id,
				fullOutputPath,
			},
			{ ownership: { kind: "conversation" }, background: true },
		);
		const doc = await tx.doc(CommandDoc, id);
		doc.state = {
			status: "starting",
			output: "",
			truncated: false,
			wall_time_seconds: 0,
			full_output_path: fullOutputPath,
		};
		return id;
	}, context);
	const watch = await api.watchDoc(CommandDoc, id, context);
	if (watch === undefined) throw new Error("Command state does not exist");
	const ready = Promise.withResolvers<void>();
	const finished = Promise.withResolvers<void>();
	ready.promise.catch(() => {});
	finished.promise.catch(() => {});
	const observe = (state: CommandState | undefined) => {
		if (state === undefined) {
			const error = new Error("Background Bash job ended before its result was received");
			ready.reject(error);
			finished.reject(error);
			return;
		}
		if (state.status !== "starting") ready.resolve();
		if (state.status === "exited" || state.status === "failed") finished.resolve();
	};
	observe(watch.value?.state);
	watch.start(async (value) => {
		observe(value?.state);
	});
	let timer: ReturnType<typeof setTimeout> | undefined;
	const deadline = new Promise<void>((resolve) => {
		timer = setTimeout(resolve, args.yield_time_ms);
	});
	try {
		await awaitWithContext(Promise.race([finished.promise, deadline]), context);
		await awaitWithContext(ready.promise, context);
		const state = await api.commit(async (tx) => {
			const doc = await tx.doc(CommandDoc, id);
			if (doc.state === undefined || doc.state.status === "starting") throw new Error("Command has not started");
			if (doc.state.status === "running") {
				doc.yielded = true;
				doc.state.wall_time_seconds = (Date.now() - startedAt) / 1000;
			}
			// A wrapper may redact only its returned content; unfinished output has not passed that policy.
			return { ...doc.state, ...(doc.state.status === "running" ? { output: "" } : {}) };
		}, context);
		return commandResult(state);
	} finally {
		clearTimeout(timer);
		await watch.stop();
	}
}

/** Dispatch before call-level wrappers, which then execute once under the command task's lifetime. */
export function executeToolInvocation(
	tool: ToolRegistration,
	args: JsonObject,
	api: ToolExecutionApi,
	context: Context,
): Promise<ToolExecutionResult> {
	const commandTool = tool as CommandTool;
	if (commandTool[commandFactory] !== undefined && args.yield_time_ms !== undefined) {
		if (
			typeof args.command !== "string" ||
			typeof args.yield_time_ms !== "number" ||
			!Number.isInteger(args.yield_time_ms) ||
			args.yield_time_ms < 0 ||
			args.yield_time_ms > 30000 ||
			(args.timeout !== undefined && typeof args.timeout !== "number")
		) {
			throw new Error("Invalid yielding Bash arguments");
		}
		return yieldBashCommand(
			{
				command: args.command,
				yield_time_ms: args.yield_time_ms,
				...(args.timeout === undefined ? {} : { timeout: args.timeout }),
			},
			api,
			context,
			commandTool,
		);
	}
	return tool.execute(args, api, context);
}

function commandResult(state: CommandState): ToolExecutionResult<CommandResult> {
	if (state.status === "starting") throw new Error("Command has not started");
	return {
		content: [{ type: "text", text: formatCommand(state) }],
		...(state.status === "failed" ? {} : { details: state }),
		...(state.status === "failed" || (state.status === "exited" && state.exit_code !== 0) ? { isError: true } : {}),
	};
}

function formatCommand(state: CommandState, command?: string, completion = false, completionPid?: number): string {
	const pid = completionPid ?? (state.status === "running" || state.status === "failed" ? state.pid : undefined);
	const header = completion ? `[BASH${pid === undefined ? "" : ` pid ${pid}`}]\n` : "";
	const status =
		state.status === "exited"
			? `Command exited with code ${state.exit_code}`
			: state.status === "failed"
				? `Command failed: ${boundOutput(state.error, { ...limits, maxBytes: 1024 }).text}`
				: `Command ${state.status}${pid === undefined ? "" : ` (pid ${pid})`}`;
	const description =
		command === undefined ? "" : `\nCommand: ${boundOutput(command, { ...limits, maxBytes: 1024 }).text}`;
	const footer = `\n${status}${description}${state.full_output_path === undefined ? "" : `\nFull output: ${state.full_output_path}`}`;
	const output = boundOutput(state.output, {
		...limits,
		maxBytes: Math.max(0, DEFAULT_MAX_BYTES - utf8ByteLength(header + footer)),
	}).text;
	return header + output + footer;
}
