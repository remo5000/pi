import { existsSync, mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
	type BashOperations,
	type BashToolOutput,
	createBashTool,
	createLocalShellOperations,
} from "../src/core/tools/bash.ts";
import {
	type BashCompletion,
	BashProcessManager,
	formatBashCompletion,
	getBashProcessManager,
} from "../src/core/tools/bash-processes.ts";
import * as outputFiles from "../src/utils/output-files.ts";

// bash.ts and bash-processes.ts: yielding must preserve process ownership, output, and exactly one terminal notification.
describe("bash", () => {
	let cwd: string;
	let processes: BashProcessManager;
	let completions: BashCompletion[];

	beforeEach(() => {
		cwd = mkdtempSync(join(tmpdir(), "pi-bash-yield-test-"));
		completions = [];
		processes = new BashProcessManager((completion) => completions.push(completion));
	});

	afterEach(async () => {
		processes.dispose();
		await processes.waitForAll();
		rmSync(cwd, { recursive: true, force: true });
		vi.useRealTimers();
		vi.restoreAllMocks();
		vi.unstubAllEnvs();
	});

	function controlledCommand() {
		let finish!: (value: { exitCode: number }) => void;
		let fail!: (error: Error) => void;
		let append!: (data: Buffer) => void;
		let processSignal: AbortSignal | undefined;
		const operations: BashOperations = {
			exec: (_command, _cwd, { onData, signal }) => {
				append = onData;
				processSignal = signal;
				return new Promise((resolve, reject) => {
					finish = resolve;
					fail = reject;
					signal?.addEventListener("abort", () => reject(new Error("aborted")), { once: true });
				});
			},
		};
		return {
			tool: createBashTool(cwd, { operations, processManager: processes }),
			append: (text: string) => append(Buffer.from(text)),
			finish: (exitCode = 0) => finish({ exitCode }),
			fail: (error: string) => fail(new Error(error)),
			get signal() {
				return processSignal;
			},
		};
	}

	it("keeps the existing arguments and adds only optional yield_time_ms", () => {
		const tool = createBashTool(cwd);
		expect(Object.keys(tool.parameters.properties)).toEqual(["command", "timeout", "yield_time_ms"]);
		expect(tool.parameters.required).toEqual(["command"]);
	});

	it("keeps Promise.allSettled blocking when yield_time_ms is omitted", async () => {
		vi.useFakeTimers();
		const first = controlledCommand();
		const second = controlledCommand();
		let settled = false;
		const execution = Promise.allSettled([
			first.tool.execute("first", { command: "first" }),
			second.tool.execute("second", { command: "second" }),
		]).then((results) => {
			settled = true;
			return results;
		});
		await vi.advanceTimersByTimeAsync(30_001);
		expect(settled).toBe(false);
		first.finish();
		await vi.advanceTimersByTimeAsync(1);
		expect(settled).toBe(false);
		second.finish(7);
		const results = await execution;
		expect(results).toEqual([
			{
				status: "fulfilled",
				value: expect.objectContaining({
					structuredContent: expect.objectContaining({ status: "exited", exit_code: 0 }),
				}),
			},
			{
				status: "fulfilled",
				value: expect.objectContaining({
					structuredContent: expect.objectContaining({ status: "exited", exit_code: 7 }),
				}),
			},
		]);
		expect(completions).toEqual([]);
	});

	it.skipIf(process.platform === "win32")(
		"runs shell syntax using the existing cwd and shell configuration",
		async () => {
			const tool = createBashTool(cwd, { processManager: processes });
			const result = await tool.execute("shell", {
				command: "printf '%s\n' \"$PWD\"; printf hello | tr a-z A-Z; printf err >&2",
				yield_time_ms: 30_000,
			});
			const output = result.structuredContent as BashToolOutput;
			expect(output.status).toBe("exited");
			if (output.status !== "exited") throw new Error("Expected a completed command");
			expect(output.exit_code).toBe(0);
			expect(output.output).toContain(cwd);
			expect(output.output).toContain("HELLO");
			expect(output.output).toContain("err");
			expect(completions).toEqual([]);
		},
	);

	it("waits for an in-progress log close before returning a result", async () => {
		const createStream = outputFiles.createOutputFileStream;
		let logClosed = false;
		let logPath: string | undefined;
		vi.spyOn(outputFiles, "createOutputFileStream").mockImplementation((prefix, extension) => {
			const file = createStream(prefix, extension);
			logPath = file.path;
			file.stream.once("finish", () => {
				logClosed = true;
			});
			const end = file.stream.end.bind(file.stream);
			vi.spyOn(file.stream, "end").mockImplementation(() => {
				setTimeout(() => end(), 30);
				return file.stream;
			});
			return file;
		});
		const tool = createBashTool(cwd, {
			processManager: processes,
			operations: {
				exec: async (_cmd, _cwd, { onData }) => {
					onData(Buffer.from("closed\n"));
					return { exitCode: 0 };
				},
			},
		});
		const result = (await tool.execute("closing", { command: "check", yield_time_ms: 0 }))
			.structuredContent as BashToolOutput;
		expect(logClosed).toBe(true);
		if (!logPath) throw new Error("No command log was created");
		expect(readFileSync(logPath, "utf8")).toBe("closed\n");
		await processes.waitForAll();
		expect(completions).toHaveLength(result.status === "running" ? 1 : 0);
	});

	it("exposes the default process owner to standalone SDK callers", async () => {
		let processSignal: AbortSignal | undefined;
		const tool = createBashTool(cwd, {
			operations: {
				exec: (_cmd, _cwd, { signal }) => {
					processSignal = signal;
					return new Promise((_resolve, reject) =>
						signal?.addEventListener("abort", () => reject(new Error("aborted")), { once: true }),
					);
				},
			},
		});
		const manager = getBashProcessManager(tool);
		if (!manager) throw new Error("Standalone Bash has no process manager");
		try {
			const result = (await tool.execute("standalone", { command: "check", yield_time_ms: 0 }))
				.structuredContent as BashToolOutput;
			expect(result.status).toBe("running");
		} finally {
			manager.dispose();
			await manager.waitForAll();
		}
		expect(processSignal?.aborted).toBe(true);
	});

	it("applies backpressure to both pipes and preserves output past the post-exit grace period", async () => {
		const operations = createLocalShellOperations("node", () => ({ shell: process.execPath, args: ["-e"] }));
		const chunks: Buffer[] = [];
		let consumers = 0;
		let maxConsumers = 0;
		const result = await operations.exec(
			"process.stdout.write('a'.repeat(262144)); process.stderr.write('b'.repeat(262144));",
			cwd,
			{
				timeout: 10,
				onData: async (data) => {
					consumers++;
					maxConsumers = Math.max(maxConsumers, consumers);
					await new Promise((resolve) => setTimeout(resolve, 120));
					chunks.push(data);
					consumers--;
				},
			},
		);
		expect(result.exitCode).toBe(0);
		expect(maxConsumers).toBeLessThanOrEqual(2);
		expect(Buffer.concat(chunks).length).toBe(524288);
		expect(consumers).toBe(0);
	});

	it("yields only after an explicitly requested positive wait", async () => {
		vi.useFakeTimers();
		const command = controlledCommand();
		let settled = false;
		const execution = command.tool.execute("deadline", { command: "check", yield_time_ms: 20 }).then((result) => {
			settled = true;
			return result;
		});
		await vi.advanceTimersByTimeAsync(19);
		expect(settled).toBe(false);
		await vi.advanceTimersByTimeAsync(1);
		expect((await execution).structuredContent).toMatchObject({ status: "running" });
		command.finish();
		await processes.waitForAll();
		expect(completions).toHaveLength(1);
	});

	it("returns a completed result without notifying when exit wins the yield race", async () => {
		const command = controlledCommand();
		const execution = command.tool.execute("fast", { command: "check", yield_time_ms: 100 });
		command.append("done\n");
		command.finish(3);
		const result = await execution;
		expect(result.isError).toBe(true);
		expect(result.structuredContent).toMatchObject({ status: "exited", exit_code: 3, output: "done\n" });
		expect(completions).toEqual([]);
	});

	it("keeps capturing output and notifies once after yielding", async () => {
		const command = controlledCommand();
		const execution = command.tool.execute("slow", { command: "check", yield_time_ms: 0 });
		command.append("before\n");
		const result = (await execution).structuredContent as BashToolOutput;
		expect(result.status).toBe("running");
		if (result.status !== "running") throw new Error("Expected a running command");
		expect(result.output).toBe("before\n");
		expect(result).not.toHaveProperty("exit_code");
		expect(readFileSync(result.full_output_path, "utf8")).toBe("before\n");
		command.append("after\n");
		command.finish(7);
		await processes.waitForAll();
		expect(readFileSync(result.full_output_path, "utf8")).toBe("before\nafter\n");
		expect(completions).toEqual([
			{
				command: "check",
				process_id: result.process_id,
				full_output_path: result.full_output_path,
				content: expect.arrayContaining([expect.objectContaining({ type: "text" })]),
				result: expect.objectContaining({
					status: "exited",
					output: "before\nafter\n",
					exit_code: 7,
				}),
			},
		]);
	});

	it("bounds completion notifications to the model-facing output limit", async () => {
		const command = controlledCommand();
		await command.tool.execute("chatty", { command: "chatty", yield_time_ms: 0 });
		command.append(`${"x".repeat(200_000)}\nlast line\n`);
		command.finish();
		await processes.waitForAll();
		expect(completions).toHaveLength(1);
		const notification = formatBashCompletion(completions[0]);
		expect(Buffer.byteLength(notification)).toBeLessThanOrEqual(50 * 1024);
		expect(notification).toContain("last line");
		expect(notification).toContain(completions[0].full_output_path);
	});

	it("bounds inline commands and failure messages in completion notifications", async () => {
		const command = controlledCommand();
		await command.tool.execute("inline", { command: "x".repeat(200_000), yield_time_ms: 0 });
		command.fail("failure".repeat(40_000));
		await processes.waitForAll();
		expect(completions).toHaveLength(1);
		const notification = formatBashCompletion(completions[0]);
		expect(Buffer.byteLength(notification)).toBeLessThanOrEqual(50 * 1024);
		expect(notification).toContain("Command failed");
		expect(notification).toContain(completions[0].full_output_path);
	});

	it("detaches the invocation abort signal only after yielding", async () => {
		const command = controlledCommand();
		const controller = new AbortController();
		await command.tool.execute("detach", { command: "check", yield_time_ms: 0 }, controller.signal);
		controller.abort();
		expect(command.signal?.aborted).toBe(false);
		command.finish();
		await processes.waitForAll();
		expect(completions).toHaveLength(1);
	});

	it("still cancels a command while the invocation is waiting", async () => {
		const command = controlledCommand();
		const controller = new AbortController();
		const execution = command.tool.execute("cancel", { command: "check" }, controller.signal);
		controller.abort();
		await expect(execution).rejects.toThrow("Command aborted");
		expect(command.signal?.aborted).toBe(true);
		expect(completions).toEqual([]);
	});

	it("kills yielded processes on disposal and suppresses their completion notifications", async () => {
		const command = controlledCommand();
		await command.tool.execute("dispose", { command: "check", yield_time_ms: 0 });
		processes.dispose();
		processes.dispose();
		await processes.waitForAll();
		expect(command.signal?.aborted).toBe(true);
		expect(completions).toEqual([]);
		await expect(command.tool.execute("late", { command: "check", yield_time_ms: 0 })).rejects.toThrow("disposed");
	});

	it("aborts current jobs without notifying and remains usable after reload", async () => {
		const first = controlledCommand();
		await first.tool.execute("old", { command: "old", yield_time_ms: 0 });
		await processes.abortAll();
		expect(first.signal?.aborted).toBe(true);
		expect(completions).toEqual([]);
		const next = controlledCommand();
		await next.tool.execute("new", { command: "new", yield_time_ms: 0 });
		next.finish();
		await processes.waitForAll();
		expect(completions).toHaveLength(1);
		expect(completions[0].command).toBe("new");
	});

	it("allows a session to subscribe to an existing tool manager and unsubscribe on reload", async () => {
		const listener = vi.fn();
		const unsubscribe = processes.onCompletion(listener);
		const first = controlledCommand();
		await first.tool.execute("subscribed", { command: "check", yield_time_ms: 0 });
		first.finish();
		await processes.waitForAll();
		expect(listener).toHaveBeenCalledTimes(1);
		unsubscribe();
		const next = controlledCommand();
		await next.tool.execute("unsubscribed", { command: "check", yield_time_ms: 0 });
		next.finish();
		await processes.waitForAll();
		expect(listener).toHaveBeenCalledTimes(1);
		expect(completions).toHaveLength(2);
	});

	it.each([false, true])(
		"reports subscriber failures without suppressing other completions (async=%s)",
		async (asyncListener) => {
			const error = new Error("subscriber failed");
			const report = vi.spyOn(console, "error").mockImplementation(() => {});
			processes.onCompletion(() => {
				if (asyncListener) return Promise.reject(error);
				throw error;
			});
			const notified = vi.fn();
			processes.onCompletion(notified);
			const command = controlledCommand();
			await command.tool.execute("subscriber", { command: "check", yield_time_ms: 0 });
			command.finish();
			await processes.waitForAll();
			expect(notified).toHaveBeenCalledTimes(1);
			expect(completions).toHaveLength(1);
			expect(report).toHaveBeenCalledWith("bash completion listener failed:", error);
		},
	);

	it("reports execution failures after yielding without inventing an exit code", async () => {
		const command = controlledCommand();
		const result = (await command.tool.execute("error", { command: "check", yield_time_ms: 0 }))
			.structuredContent as BashToolOutput;
		if (result.status !== "running") throw new Error("Expected a running command");
		command.fail("timeout:5");
		await processes.waitForAll();
		expect(completions).toEqual([
			{
				command: "check",
				process_id: result.process_id,
				full_output_path: result.full_output_path,
				error: "Command timed out after 5 seconds",
			},
		]);
	});

	it("stops streaming tool updates when it returns a running result", async () => {
		const command = controlledCommand();
		const updates = vi.fn();
		await command.tool.execute("updates", { command: "check", yield_time_ms: 0 }, undefined, updates);
		const count = updates.mock.calls.length;
		command.append("late\n");
		command.finish();
		await processes.waitForAll();
		expect(updates).toHaveBeenCalledTimes(count);
	});

	it("keeps a bounded running snapshot and preserves the complete private log", async () => {
		const command = controlledCommand();
		const execution = command.tool.execute("large", { command: "chatty", yield_time_ms: 0 });
		command.append("€".repeat(400_000));
		const output = (await execution).structuredContent as BashToolOutput;
		if (output.status !== "running") throw new Error("Expected a running command");
		expect(output.truncated).toBe(true);
		expect(output.output).not.toContain("�");
		expect(Buffer.byteLength(output.output)).toBeLessThanOrEqual(50 * 1024);
		expect(readFileSync(output.full_output_path, "utf8")).toBe("€".repeat(400_000));
		if (process.platform !== "win32") expect(statSync(output.full_output_path).mode & 0o777).toBe(0o600);
		command.finish();
		await processes.waitForAll();
	});

	it.each([
		{ yield_time_ms: -1 },
		{ yield_time_ms: 30_001 },
		{ yield_time_ms: Number.NaN },
		{ yield_time_ms: 1.5 },
		{ yield_time_ms: Number.POSITIVE_INFINITY },
	])("rejects invalid limits before starting a process: %j", async (limits) => {
		const exec = vi.fn();
		const tool = createBashTool(cwd, { processManager: processes, operations: { exec } });
		await expect(tool.execute("invalid", { command: "check", ...limits })).rejects.toThrow("Invalid");
		expect(exec).not.toHaveBeenCalled();
	});

	it("does not start commands whose signal is already aborted", async () => {
		const controller = new AbortController();
		controller.abort();
		const exec = vi.fn();
		const tool = createBashTool(cwd, { processManager: processes, operations: { exec } });
		await expect(tool.execute("aborted", { command: "check" }, controller.signal)).rejects.toThrow("aborted");
		expect(exec).not.toHaveBeenCalled();
	});

	it.skipIf(process.platform === "win32")("keeps the hard timeout active after yielding", async () => {
		const tool = createBashTool(cwd, { processManager: processes });
		const output = (await tool.execute("timeout", { command: "sleep 30", yield_time_ms: 0, timeout: 0.05 }))
			.structuredContent as BashToolOutput;
		expect(output.status).toBe("running");
		if (output.status !== "running") throw new Error("Expected a running command");
		await processes.waitForAll();
		expect(completions).toEqual([
			{
				command: "sleep 30",
				process_id: output.process_id,
				full_output_path: output.full_output_path,
				error: "Command timed out after 0.05 seconds",
			},
		]);
		expect(existsSync(output.full_output_path)).toBe(true);
	});
});
