// Durable Bash: explicit yielding must detach execution, not its timeout, output, or completion state.
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Context, JsonValue } from "@earendil-works/chord";
import { awaitWithContext } from "@earendil-works/chord/context";
import { fauxAssistantMessage, fauxText, fauxToolCall, type ToolResultMessage } from "@earendil-works/pi-ai";
import { type Conversation, type Harness, MemoryStorage, ToolResultEntry, ToolTask } from "@earendil-works/pi-durable";
import { afterEach, describe, expect, it } from "vitest";
import {
	type ExecutionError,
	getOrThrow,
	ok,
	type Result,
	type ShellExecOptions,
	type ShellExecResult,
} from "../src/env/index.ts";
import { NodeExecutionEnv } from "../src/env/node.ts";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { createBashTool } from "../src/tools/index.ts";
import { allEntries, type ChatSetup, chatSetup, openChat, textOf, waitFor } from "./chat-support.ts";
import { addHooks, addTool } from "./harness-support.ts";
import { context } from "./session-support.ts";
import { deferred, settled } from "./task-support.ts";

const resources: { harness: Harness; directory: string }[] = [];
afterEach(async () => {
	for (const { harness, directory } of resources.splice(0)) {
		await harness.close(context);
		await rm(directory, { recursive: true, force: true });
	}
});

class ControlledEnv extends NodeExecutionEnv {
	readonly started = deferred();
	readonly start = deferred();
	readonly finish = deferred<{ output: string; exitCode: number }>();
	calls = 0;
	aborted = false;
	options: ShellExecOptions | undefined;
	reportedPid: number | undefined = 4242;

	override async exec(
		_command: string | readonly string[],
		options: ShellExecOptions | undefined,
		callContext: Context,
	): Promise<Result<ShellExecResult, ExecutionError>> {
		this.calls++;
		this.options = options;
		this.started.resolve();
		callContext.abortSignal?.addEventListener(
			"abort",
			() => {
				this.aborted = true;
			},
			{ once: true },
		);
		await awaitWithContext(this.start.promise, callContext);
		await options?.onSpawn?.(this.reportedPid, callContext);
		const finished = await awaitWithContext(this.finish.promise, callContext);
		if (options?.spill?.path !== undefined) {
			getOrThrow(await this.appendFile(options.spill.path, finished.output, callContext));
		}
		options?.onOutput?.(finished.output, callContext, { stream: "stdout" });
		return ok({
			exitCode: finished.exitCode,
			...(options?.spill?.path === undefined ? {} : { spillPath: options.spill.path }),
		});
	}
}

async function open(
	args: Record<string, JsonValue>,
	real = false,
	persistent = false,
	configure?: (setup: ChatSetup, env: NodeExecutionEnv) => void,
) {
	const directory = await mkdtemp(join(tmpdir(), "pi-durable-yield-"));
	const env = real ? new NodeExecutionEnv({ cwd: directory }) : new ControlledEnv({ cwd: directory });
	const setup = chatSetup();
	addTool(setup.registry, createBashTool());
	configure?.(setup, env);
	setup.faux.setResponses([
		fauxAssistantMessage([fauxToolCall("bash", args, { id: "bash-call" })], { stopReason: "toolUse" }),
		...Array.from({ length: 8 }, () => fauxAssistantMessage([fauxText("done")])),
	]);
	const storagePath = join(directory, "session.sqlite");
	const storage = persistent ? await openNodeSqliteStorage(storagePath) : new MemoryStorage();
	const { harness, root } = await openChat(storage, setup, { env });
	resources.push({ harness, directory });
	const submission = await root.submit({ type: "input", content: "run command" }, context);
	return { harness, root, env, setup, storagePath, submission };
}

async function result(root: Conversation): Promise<ToolResultMessage | undefined> {
	return (await allEntries(root)).find((entry) => ToolResultEntry.is(entry))?.model?.[0] as
		| ToolResultMessage
		| undefined;
}

async function completions(root: Conversation): Promise<string[]> {
	return (await allEntries(root)).flatMap((entry) => {
		const message = entry.model?.[0];
		const text = textOf(message);
		return message?.role === "user" && text?.startsWith("[BASH") ? [text] : [];
	});
}

function controlled(env: NodeExecutionEnv): ControlledEnv {
	if (!(env instanceof ControlledEnv)) throw new Error("Expected controlled environment");
	return env;
}

describe("Durable Bash yielding", () => {
	it("blocks when yielding is omitted and records an exited result", async () => {
		const { env, root, submission } = await open({ command: "command" });
		const process = controlled(env);
		process.start.resolve();
		await process.started.promise;
		expect(await settled(submission.wait(context))).toBe(false);
		process.finish.resolve({ output: "finished\n", exitCode: 0 });
		expect((await submission.wait(context)).status).toBe("done");
		expect((await result(root))?.details).toMatchObject({ status: "exited", exit_code: 0, output: "finished\n" });
		expect(await completions(root)).toEqual([]);
	});

	it("zero wait yields only after real startup and keeps the command alive", async () => {
		const { env, root, submission, harness, setup } = await open({ command: "command", yield_time_ms: 0 });
		const process = controlled(env);
		await process.started.promise;
		expect(await settled(submission.wait(context))).toBe(false);
		process.start.resolve();
		await waitFor(async () => (await result(root)) !== undefined);
		const initial = (await result(root))!;
		expect(initial.isError).toBe(false);
		expect(initial.details).toMatchObject({ status: "running", pid: 4242, full_output_path: expect.any(String) });
		expect(initial.details).not.toHaveProperty("exit_code");
		expect(JSON.stringify(initial)).not.toMatch(/command_id|process_id/);
		await submission.wait(context);
		await root.waitForIdle(context);
		expect(process.aborted).toBe(false);
		expect((await harness.inspect(context)).tasks.some(({ record }) => record.background)).toBe(true);
		process.finish.resolve({ output: "finished after yielding\n", exitCode: 0 });
		await waitFor(async () => (await completions(root)).length === 1);
		await harness.waitForIdle(context);
		expect((await completions(root))[0]).toContain("finished after yielding");
		expect((await completions(root))[0]).toContain("pid 4242");
		const path = (initial.details as { full_output_path: string }).full_output_path;
		expect(await readFile(path, "utf8")).toBe("finished after yielding\n");
		expect(setup.reports).toEqual([]);
	});

	it("returns exited without a second notification when completion beats the yield deadline", async () => {
		const { env, root, submission } = await open({ command: "command", yield_time_ms: 1000 });
		const process = controlled(env);
		process.start.resolve();
		process.finish.resolve({ output: "quick\n", exitCode: 0 });
		await submission.wait(context);
		expect((await result(root))?.details).toMatchObject({ status: "exited", exit_code: 0, output: "quick\n" });
		expect(await completions(root)).toEqual([]);
	});

	it("does not replay a yielded command after closing and reopening", async () => {
		const { env, submission, harness, storagePath, setup } = await open(
			{ command: "command", yield_time_ms: 0 },
			false,
			true,
		);
		const process = controlled(env);
		process.start.resolve();
		await submission.wait(context);
		await harness.close(context);
		expect(process.aborted).toBe(true);
		const reopened = await openChat(await openNodeSqliteStorage(storagePath), setup, { env });
		resources.push({ harness: reopened.harness, directory: env.cwd });
		reopened.harness.resume();
		await waitFor(async () => (await completions(reopened.root)).length === 1);
		expect(process.calls).toBe(1);
		expect((await completions(reopened.root))[0]).toMatch(/interrupted/i);
		expect((await result(reopened.root))?.details).toMatchObject({ status: "running" });
	});

	it.each([-1, 30001, 0.5, null])(
		"rejects invalid yield_time_ms %s before starting any command",
		async (yieldTime) => {
			const { root, env, submission } = await open({ command: "command", yield_time_ms: yieldTime });
			await submission.wait(context);
			expect((await result(root))?.isError).toBe(true);
			expect(controlled(env).calls).toBe(0);
		},
	);

	it("does not claim startup for a backend without a startup capability", async () => {
		const { root, env, submission } = await open(
			{ command: "command", yield_time_ms: 0 },
			false,
			false,
			(_setup, env) => {
				Object.defineProperty(env, "reportsSpawn", { value: false });
			},
		);
		await submission.wait(context);
		expect((await result(root))?.isError).toBe(true);
		expect(controlled(env).calls).toBe(0);
	});

	it("rejects an oversized backend log path before launching a command", async () => {
		const { root, env, submission } = await open(
			{ command: "command", yield_time_ms: 0 },
			false,
			false,
			(_setup, env) => {
				env.createTempFile = async () => ok(`log://${"x".repeat(60 * 1024)}`);
			},
		);
		const process = controlled(env);
		process.start.resolve();
		await submission.wait(context);
		expect((await result(root))?.isError).toBe(true);
		expect(process.calls).toBe(0);
	});

	it("can report actual startup without a numeric PID", async () => {
		const { root, env, submission } = await open(
			{ command: "command", yield_time_ms: 0 },
			false,
			false,
			(_setup, env) => {
				controlled(env).reportedPid = undefined;
			},
		);
		const process = controlled(env);
		process.start.resolve();
		await submission.wait(context);
		expect((await result(root))?.details).toMatchObject({ status: "running" });
		expect((await result(root))?.details).not.toHaveProperty("pid");
		process.finish.resolve({ output: "done\n", exitCode: 0 });
		await waitFor(async () => (await completions(root)).length === 1);
		expect((await completions(root))[0]).toMatch(/^\[BASH\]/);
	});

	it("cancels startup when the original tool is aborted before yielding", async () => {
		const { root, env, harness } = await open({ command: "command", yield_time_ms: 0 });
		const process = controlled(env);
		await process.started.promise;
		await root.abort(context);
		await waitFor(() => process.aborted);
		await waitFor(async () => (await harness.inspect(context)).tasks.length === 0);
		expect(await completions(root)).toEqual([]);
	});

	it("ordinary conversation abort leaves yielded execution alive", async () => {
		const { root, env, submission } = await open({ command: "command", yield_time_ms: 0 });
		const process = controlled(env);
		process.start.resolve();
		await submission.wait(context);
		await root.abort(context);
		expect(process.aborted).toBe(false);
		process.finish.resolve({ output: "late\n", exitCode: 0 });
		await waitFor(async () => (await completions(root)).length === 1);
	});

	it("explicit background abort cancels and joins yielded execution without a completion", async () => {
		const { root, env, submission, harness } = await open({ command: "command", yield_time_ms: 0 });
		const process = controlled(env);
		process.start.resolve();
		await submission.wait(context);
		await root.abort(context, { background: true });
		expect(process.aborted).toBe(true);
		expect((await harness.inspect(context)).tasks).toHaveLength(0);
		expect(await completions(root)).toEqual([]);
	});

	it("requires the actual tool result to authorize detaching, not just a yielded marker", async () => {
		const hookReached = deferred();
		const releaseHook = deferred();
		const { root, env, harness } = await open({ command: "command", yield_time_ms: 0 }, false, false, (setup) => {
			addHooks(setup.registry, ToolTask, {
				afterTool: async (_call, result) => {
					hookReached.resolve();
					await releaseHook.promise;
					return { ...result, isError: true };
				},
			});
		});
		const process = controlled(env);
		process.start.resolve();
		await hookReached.promise;
		expect(process.aborted).toBe(false);
		releaseHook.resolve();
		await waitFor(() => process.aborted);
		await waitFor(async () => (await harness.inspect(context)).tasks.length === 0);
		expect(await completions(root)).toEqual([]);
	});

	it("keeps environment values injected by prepare out of durable command input", async () => {
		const injected = "ephemeral-prepare-fixture";
		const { root, env, submission, harness } = await open(
			{ command: "command", yield_time_ms: 0 },
			false,
			false,
			(setup) => {
				addTool(
					setup.registry,
					createBashTool({
						prepare: (execution) => {
							execution.env = { TEST_PREPARED: injected };
							execution.command = `rewritten-${injected}`;
						},
					}),
				);
			},
		);
		const process = controlled(env);
		process.start.resolve();
		await submission.wait(context);
		expect(process.options?.env).toEqual({ TEST_PREPARED: injected });
		const tasks = await harness.commit((tx) => tx.scanTasks({ conversationId: root.id }, 100), context);
		expect(JSON.stringify(tasks.items.map((task) => task.input))).not.toContain(injected);
		process.finish.resolve({ output: "done\n", exitCode: 0 });
		await waitFor(async () => (await completions(root)).length === 1);
	});

	it("bounds an entire completion including long command and Unicode output", async () => {
		const { root, env, submission } = await open({ command: "c".repeat(100000), yield_time_ms: 0 });
		const process = controlled(env);
		process.start.resolve();
		await submission.wait(context);
		process.finish.resolve({ output: "é".repeat(100000), exitCode: 7 });
		await waitFor(async () => (await completions(root)).length === 1);
		const text = (await completions(root))[0]!;
		expect(Buffer.byteLength(text)).toBeLessThanOrEqual(50 * 1024);
		expect(text).toContain("Command exited with code 7");
		expect(text).toContain("Full output:");
	});

	it("resumes a committed report without submitting the completion twice", async () => {
		const { root, env, submission, harness, setup, storagePath } = await open(
			{ command: "command", yield_time_ms: 0 },
			false,
			true,
		);
		const process = controlled(env);
		process.start.resolve();
		await submission.wait(context);
		let closing: Promise<void> | undefined;
		const stop = harness.subscribeCommits((publication) => {
			if (
				closing === undefined &&
				publication.changes.some(
					(change) => change.type === "submission" && change.value.requestId?.startsWith("bash-completion:"),
				)
			) {
				closing = harness.close(context);
			}
		});
		try {
			process.finish.resolve({ output: "completed once\n", exitCode: 0 });
			await waitFor(() => closing !== undefined);
			await closing;
		} finally {
			stop();
		}
		const reopenedStorage = await openNodeSqliteStorage(storagePath);
		const reopened = await openChat(reopenedStorage, setup, { env });
		resources.push({ harness: reopened.harness, directory: env.cwd });
		reopened.harness.resume();
		await waitFor(async () => (await reopened.harness.inspect(context)).tasks.length === 0);
		const records = await reopenedStorage.scanSubmissions({ conversationId: root.id }, 100, undefined, context);
		expect(records.items.filter((record) => record.requestId?.startsWith("bash-completion:"))).toHaveLength(1);
		expect(await completions(reopened.root)).toHaveLength(1);
		expect(process.calls).toBe(1);
	});

	it("preserves a hard timeout after yielding with a real local process", async () => {
		const { root, submission } = await open(
			{ command: "printf started; sleep 10", timeout: 0.15, yield_time_ms: 0 },
			true,
		);
		await submission.wait(context);
		expect((await result(root))?.details).toMatchObject({ status: "running", pid: expect.any(Number) });
		await waitFor(async () => (await completions(root)).length === 1);
		expect((await completions(root))[0]).toMatch(/timed out/i);
	});
});
