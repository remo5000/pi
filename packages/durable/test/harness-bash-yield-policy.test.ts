import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Context } from "@earendil-works/chord";
import { awaitWithContext } from "@earendil-works/chord/context";
import { fauxAssistantMessage, fauxText, fauxToolCall, type ToolResultMessage } from "@earendil-works/pi-ai";
import { defineExtension, type Harness, MemoryStorage, ToolResultEntry, ToolTask } from "@earendil-works/pi-durable";
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
import { createBashTool } from "../src/tools/bash.ts";
import { CommandDoc } from "../src/tools/bash-command.ts";
import type { ConversationId, EntryId, TaskId } from "../src/types.ts";
import { allEntries, chatSetup, openChat, textOf, waitFor } from "./chat-support.ts";
import { addHooks } from "./harness-support.ts";
import { context } from "./session-support.ts";
import { deferred } from "./task-support.ts";

const resources: { harness: Harness; directory: string }[] = [];
afterEach(async () => {
	for (const { harness, directory } of resources.splice(0)) {
		await harness.close(context);
		await rm(directory, { recursive: true, force: true });
	}
});

class ScriptedEnv extends NodeExecutionEnv {
	readonly commands: string[] = [];
	readonly launched = deferred();
	readonly finish = deferred();
	readonly partialSent = deferred();
	partialOutput?: string;
	aborted = false;
	exitCode = 0;

	override async exec(
		command: string | readonly string[],
		options: ShellExecOptions | undefined,
		callContext: Context,
	): Promise<Result<ShellExecResult, ExecutionError>> {
		this.commands.push(typeof command === "string" ? command : command.join(" "));
		this.launched.resolve();
		callContext.abortSignal?.addEventListener(
			"abort",
			() => {
				this.aborted = true;
			},
			{ once: true },
		);
		await options?.onSpawn?.(4242, callContext);
		if (this.partialOutput !== undefined) {
			options?.onOutput?.(this.partialOutput, callContext, { stream: "stdout" });
			this.partialSent.resolve();
		}
		await awaitWithContext(this.finish.promise, callContext);
		const output = "secret payload\n";
		if (options?.spill?.path !== undefined)
			getOrThrow(await this.appendFile(options.spill.path, `${this.partialOutput ?? ""}${output}`, callContext));
		options?.onOutput?.(output, callContext, { stream: "stdout" });
		return ok({
			exitCode: this.exitCode,
			...(options?.spill?.path === undefined ? {} : { spillPath: options.spill.path }),
		});
	}
}

class FailingCallerReadStorage extends MemoryStorage {
	callerId?: number;
	failNextCallerRead = false;
	failedCallerRead = false;

	override async task(...args: Parameters<MemoryStorage["task"]>) {
		if (this.failNextCallerRead && args[0] === this.callerId) {
			this.failNextCallerRead = false;
			this.failedCallerRead = true;
			throw new Error("Caller task read failed");
		}
		return super.task(...args);
	}
}

class FailingCallerEntryStorage extends MemoryStorage {
	callerEntryId?: EntryId;
	failedCallerEntryRead = false;

	override entry(id: EntryId, callContext: Context): ReturnType<MemoryStorage["entry"]>;
	override entry(
		conversationId: ConversationId,
		id: EntryId,
		callContext: Context,
	): ReturnType<MemoryStorage["entry"]>;
	override entry(
		idOrConversationId: EntryId | ConversationId,
		idOrContext: EntryId | Context,
		callContext?: Context,
	): ReturnType<MemoryStorage["entry"]> {
		if (callContext !== undefined && idOrContext === this.callerEntryId) {
			this.callerEntryId = undefined;
			this.failedCallerEntryRead = true;
			return Promise.reject(new Error("Caller result entry read failed"));
		}
		return callContext === undefined
			? super.entry(idOrConversationId as EntryId, idOrContext as Context)
			: super.entry(idOrConversationId as ConversationId, idOrContext as EntryId, callContext);
	}
}

async function toolResult(root: Awaited<ReturnType<typeof openChat>>["root"]): Promise<ToolResultMessage | undefined> {
	return (await allEntries(root)).find((entry) => ToolResultEntry.is(entry))?.model?.[0] as
		| ToolResultMessage
		| undefined;
}

function occurrences(text: string, fragment: string): number {
	return text.split(fragment).length - 1;
}

describe("yielded Bash wrapper policy", () => {
	it("runs an output-redacting spread wrapper and each command prefix exactly once on the background execution", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-bash-wrapper-"));
		const env = new ScriptedEnv({ cwd: directory });
		const setup = chatSetup();
		const base = createBashTool({ commandPrefix: "printf 'factory-prefix\\n'" });
		let wrapperCalls = 0;
		const wrapped = {
			...base,
			async execute(
				args: Parameters<typeof base.execute>[0],
				api: Parameters<typeof base.execute>[1],
				callContext: Context,
			) {
				wrapperCalls++;
				return base.execute(
					{ ...args, command: `printf 'wrapper-prefix\\n'\n${args.command}` },
					{
						...api,
						output: (chunk: string | Uint8Array, skipped) =>
							api.output(
								(typeof chunk === "string" ? chunk : new TextDecoder().decode(chunk)).replaceAll(
									"secret",
									"[REDACTED]",
								),
								skipped,
							),
					},
					callContext,
				);
			},
		};
		setup.registry.install(defineExtension({ name: "bash-policy", tools: [wrapped] }));
		setup.faux.setResponses([
			fauxAssistantMessage([fauxToolCall("bash", { command: "printf 'original\\n'", yield_time_ms: 0 })], {
				stopReason: "toolUse",
			}),
			...Array.from({ length: 4 }, () => fauxAssistantMessage([fauxText("done")])),
		]);
		const { harness, root } = await openChat(new MemoryStorage(), setup, { env });
		resources.push({ harness, directory });

		await root.submit({ type: "input", content: "run" }, context);
		await env.launched.promise;
		await waitFor(async () => (await toolResult(root))?.details !== undefined);
		env.finish.resolve();
		await waitFor(async () =>
			(await allEntries(root)).some(
				(entry) => entry.model?.[0]?.role === "user" && textOf(entry.model[0])?.startsWith("[BASH"),
			),
		);

		expect(wrapperCalls).toBe(1);
		expect(env.commands).toHaveLength(1);
		expect(occurrences(env.commands[0]!, "factory-prefix")).toBe(1);
		expect(occurrences(env.commands[0]!, "wrapper-prefix")).toBe(1);
		const messages = (await allEntries(root)).flatMap((entry) => entry.model ?? []);
		const completion = messages.find((message) => message.role === "user" && textOf(message)?.startsWith("[BASH"));
		expect(textOf(completion)).toContain("[REDACTED] payload");
		expect(textOf(completion)).not.toContain("secret payload");
	});

	it("does not expose raw blocking output in details after an output-redacting wrapper", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-bash-blocking-policy-"));
		const env = new ScriptedEnv({ cwd: directory });
		const setup = chatSetup();
		const base = createBashTool();
		const wrapped = {
			...base,
			execute(
				args: Parameters<typeof base.execute>[0],
				api: Parameters<typeof base.execute>[1],
				callContext: Context,
			) {
				return base.execute(
					args,
					{
						...api,
						output: (chunk: string | Uint8Array, skipped) =>
							api.output(
								(typeof chunk === "string" ? chunk : new TextDecoder().decode(chunk)).replaceAll(
									"secret",
									"[REDACTED]",
								),
								skipped,
							),
					},
					callContext,
				);
			},
		};
		setup.registry.install(defineExtension({ name: "bash-policy", tools: [wrapped] }));
		setup.faux.setResponses([
			fauxAssistantMessage([fauxToolCall("bash", { command: "printf body" })], { stopReason: "toolUse" }),
			...Array.from({ length: 4 }, () => fauxAssistantMessage([fauxText("done")])),
		]);
		const { harness, root } = await openChat(new MemoryStorage(), setup, { env });
		resources.push({ harness, directory });

		await root.submit({ type: "input", content: "run" }, context);
		await env.launched.promise;
		env.finish.resolve();
		await waitFor(async () => (await toolResult(root)) !== undefined);
		const result = await toolResult(root);
		expect(textOf(result)).toContain("[REDACTED] payload");
		expect(textOf(result)).not.toContain("secret payload");
		expect(result?.details).toMatchObject({ output: "[REDACTED] payload\n" });
	});

	it("fails closed when an API-cloning wrapper drops the managed Bash marker", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-bash-cloned-api-"));
		const env = new ScriptedEnv({ cwd: directory });
		const setup = chatSetup();
		const base = createBashTool();
		let wrapperCalls = 0;
		const wrapped = {
			...base,
			execute(
				args: Parameters<typeof base.execute>[0],
				api: Parameters<typeof base.execute>[1],
				callContext: Context,
			) {
				wrapperCalls++;
				// Bound the regression if the first job incorrectly admits a second job.
				if (wrapperCalls > 1) throw new Error("Nested Bash job executed");
				return base.execute(args, Object.fromEntries(Object.entries(api)) as typeof api, callContext);
			},
		};
		setup.registry.install(defineExtension({ name: "bash-policy", tools: [wrapped] }));
		setup.faux.setResponses([
			fauxAssistantMessage([fauxToolCall("bash", { command: "printf body", yield_time_ms: 0 })], {
				stopReason: "toolUse",
			}),
			...Array.from({ length: 4 }, () => fauxAssistantMessage([fauxText("done")])),
		]);
		const { harness, root } = await openChat(new MemoryStorage(), setup, { env });
		resources.push({ harness, directory });
		const commandIds = new Set<number>();
		const unsubscribe = harness.subscribeCommits((publication) => {
			for (const change of publication.changes) {
				if (change.type === "task" && change.value.kind === "pi.bash-command") commandIds.add(change.value.id);
			}
		});
		try {
			await root.submit({ type: "input", content: "run" }, context);
			await waitFor(async () => (await toolResult(root)) !== undefined);
			expect(commandIds.size).toBe(1);
			expect(wrapperCalls).toBe(1);
			expect(env.commands).toEqual([]);
			expect((await toolResult(root))?.isError).toBe(true);
		} finally {
			unsubscribe();
			env.finish.resolve();
		}
	});

	it("cancels a running command if monitoring the caller fails", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-bash-monitor-failure-"));
		const env = new ScriptedEnv({ cwd: directory });
		const storage = new FailingCallerReadStorage();
		const setup = chatSetup();
		setup.registry.install(defineExtension({ name: "bash-policy", tools: [createBashTool()] }));
		const reachedHook = deferred();
		const releaseHook = deferred();
		addHooks(setup.registry, ToolTask, {
			afterTool: async (_call, result) => {
				reachedHook.resolve();
				await releaseHook.promise;
				return result;
			},
		});
		setup.faux.setResponses([
			fauxAssistantMessage([fauxToolCall("bash", { command: "printf body", yield_time_ms: 0 })], {
				stopReason: "toolUse",
			}),
			...Array.from({ length: 4 }, () => fauxAssistantMessage([fauxText("done")])),
		]);
		const { harness, root } = await openChat(storage, setup, { env });
		resources.push({ harness, directory });
		const unsubscribe = harness.subscribeCommits((publication) => {
			for (const change of publication.changes) {
				if (change.type === "task" && change.value.kind === "pi.tool") storage.callerId = change.value.id;
			}
		});
		try {
			await root.submit({ type: "input", content: "run" }, context);
			await reachedHook.promise;
			storage.failNextCallerRead = true;
			await waitFor(() => storage.failedCallerRead);
			await waitFor(() => env.aborted, 500);
			expect(env.commands).toHaveLength(1);
		} finally {
			unsubscribe();
			releaseHook.resolve();
			env.finish.resolve();
		}
	});

	it("rejects a retained startup callback after the command task has settled", async () => {
		class RetainedStartupEnv extends ScriptedEnv {
			onSpawn?: NonNullable<ShellExecOptions["onSpawn"]>;

			override exec(
				command: string | readonly string[],
				options: ShellExecOptions | undefined,
				callContext: Context,
			): Promise<Result<ShellExecResult, ExecutionError>> {
				this.onSpawn = options?.onSpawn;
				return super.exec(command, options, callContext);
			}
		}
		const directory = await mkdtemp(join(tmpdir(), "pi-bash-late-startup-"));
		const env = new RetainedStartupEnv({ cwd: directory });
		const setup = chatSetup();
		setup.registry.install(defineExtension({ name: "bash-policy", tools: [createBashTool()] }));
		setup.faux.setResponses([
			fauxAssistantMessage([fauxToolCall("bash", { command: "printf body", yield_time_ms: 0 })], {
				stopReason: "toolUse",
			}),
			...Array.from({ length: 4 }, () => fauxAssistantMessage([fauxText("done")])),
		]);
		const { harness, root } = await openChat(new MemoryStorage(), setup, { env });
		resources.push({ harness, directory });

		await root.submit({ type: "input", content: "run" }, context);
		await env.launched.promise;
		await waitFor(async () => (await toolResult(root))?.details !== undefined);
		env.finish.resolve();
		await waitFor(async () =>
			(await harness.inspect(context)).tasks.every(({ record }) => record.kind !== "pi.bash-command"),
		);
		const onSpawn = env.onSpawn;
		if (onSpawn === undefined) throw new Error("The environment did not receive a startup callback");
		let timer: ReturnType<typeof setTimeout> | undefined;
		try {
			const outcome = await Promise.race([
				Promise.resolve()
					.then(() => onSpawn(4343, context))
					.then(
						() => "resolved",
						(error: unknown) => (error instanceof Error ? error.message : String(error)),
					),
				new Promise<"pending">((resolve) => {
					timer = setTimeout(() => resolve("pending"), 200);
				}),
			]);
			expect(outcome).toMatch(/ended/i);
		} finally {
			clearTimeout(timer);
		}
	});

	it("settles the command job when reading the caller's result entry rejects", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-bash-caller-entry-"));
		const env = new ScriptedEnv({ cwd: directory });
		const storage = new FailingCallerEntryStorage();
		const setup = chatSetup();
		setup.registry.install(defineExtension({ name: "bash-policy", tools: [createBashTool()] }));
		setup.faux.setResponses([
			fauxAssistantMessage([fauxToolCall("bash", { command: "printf body", yield_time_ms: 0 })], {
				stopReason: "toolUse",
			}),
			...Array.from({ length: 4 }, () => fauxAssistantMessage([fauxText("done")])),
		]);
		const { harness, root } = await openChat(storage, setup, { env });
		resources.push({ harness, directory });
		const unsubscribe = harness.subscribeCommits((publication) => {
			for (const change of publication.changes) {
				if (change.type === "entry" && change.value.kind === ToolResultEntry.kind)
					storage.callerEntryId = change.value.id;
			}
		});
		try {
			await root.submit({ type: "input", content: "run" }, context);
			await env.launched.promise;
			await waitFor(() => storage.failedCallerEntryRead);
			await waitFor(() => env.aborted);
			await waitFor(
				async () => (await harness.inspect(context)).tasks.every(({ record }) => record.kind !== "pi.bash-command"),
				750,
			);
			expect(env.commands).toHaveLength(1);
		} finally {
			unsubscribe();
			env.finish.resolve();
		}
	});

	it("does not publish raw blocking details when a nonzero command is aborted", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-bash-aborted-details-"));
		const env = new ScriptedEnv({ cwd: directory });
		env.exitCode = 7;
		const setup = chatSetup();
		const base = createBashTool();
		const detailsPublished = deferred();
		const releaseDetails = deferred();
		const wrapped = {
			...base,
			execute(
				args: Parameters<typeof base.execute>[0],
				api: Parameters<typeof base.execute>[1],
				callContext: Context,
			) {
				return base.execute(
					args,
					{
						...api,
						output: (chunk: string | Uint8Array, skipped) =>
							api.output(
								(typeof chunk === "string" ? chunk : new TextDecoder().decode(chunk)).replaceAll(
									"secret",
									"[REDACTED]",
								),
								skipped,
							),
						async details(value, detailsContext) {
							await api.details(value, detailsContext);
							detailsPublished.resolve();
							await awaitWithContext(releaseDetails.promise, detailsContext);
						},
					},
					callContext,
				);
			},
		};
		setup.registry.install(defineExtension({ name: "bash-policy", tools: [wrapped] }));
		setup.faux.setResponses([
			fauxAssistantMessage([fauxToolCall("bash", { command: "printf body" })], { stopReason: "toolUse" }),
			...Array.from({ length: 4 }, () => fauxAssistantMessage([fauxText("done")])),
		]);
		const { harness, root } = await openChat(new MemoryStorage(), setup, { env });
		resources.push({ harness, directory });
		let callerId: TaskId | undefined;
		const unsubscribe = harness.subscribeCommits((publication) => {
			for (const change of publication.changes) {
				if (change.type === "task" && change.value.kind === "pi.tool") callerId = change.value.id;
			}
		});
		try {
			await root.submit({ type: "input", content: "run" }, context);
			await env.launched.promise;
			env.finish.resolve();
			await detailsPublished.promise;
			if (callerId === undefined) throw new Error("The Bash tool task was not admitted");
			await harness.abortTask(callerId, context);
			await waitFor(async () => (await toolResult(root)) !== undefined);
			const result = await toolResult(root);
			expect(result?.isError).toBe(true);
			expect(textOf(result)).not.toContain("secret payload");
			expect(result?.details).toMatchObject({ output: "[REDACTED] payload\n" });
		} finally {
			unsubscribe();
			releaseDetails.resolve();
			env.finish.resolve();
		}
	});

	it("honors a background wrapper's returned content when reporting completion", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-bash-result-policy-"));
		const env = new ScriptedEnv({ cwd: directory });
		env.partialOutput = "secret prelude\n";
		const setup = chatSetup();
		const base = createBashTool();
		const wrapped = {
			...base,
			async execute(
				args: Parameters<typeof base.execute>[0],
				api: Parameters<typeof base.execute>[1],
				callContext: Context,
			) {
				const result = await base.execute(args, api, callContext);
				return {
					...result,
					content: [{ type: "text" as const, text: "[REDACTED] prelude\n[REDACTED] payload\n" }],
				};
			},
		};
		setup.registry.install(defineExtension({ name: "bash-policy", tools: [wrapped] }));
		setup.faux.setResponses([
			fauxAssistantMessage([fauxToolCall("bash", { command: "printf body", yield_time_ms: 500 })], {
				stopReason: "toolUse",
			}),
			...Array.from({ length: 4 }, () => fauxAssistantMessage([fauxText("done")])),
		]);
		const { harness, root } = await openChat(new MemoryStorage(), setup, { env });
		resources.push({ harness, directory });

		await root.submit({ type: "input", content: "run" }, context);
		await env.launched.promise;
		await env.partialSent.promise;
		await waitFor(async () => (await toolResult(root))?.details !== undefined);
		const running = await toolResult(root);
		expect(running?.details).toMatchObject({ status: "running", output: "" });
		expect(textOf(running)).not.toContain("secret prelude");
		const path = (running?.details as { full_output_path: string }).full_output_path;
		env.finish.resolve();
		await waitFor(async () =>
			(await allEntries(root)).some(
				(entry) => entry.model?.[0]?.role === "user" && textOf(entry.model[0])?.startsWith("[BASH"),
			),
		);
		const completion = (await allEntries(root))
			.flatMap((entry) => entry.model ?? [])
			.find((message) => message.role === "user" && textOf(message)?.startsWith("[BASH"));
		expect(textOf(completion)).toContain("[REDACTED] prelude\n[REDACTED] payload");
		expect(textOf(completion)).not.toContain("secret prelude");
		expect(textOf(completion)).not.toContain("secret payload");
		expect(await readFile(path, "utf8")).toBe("secret prelude\nsecret payload\n");
	});

	it("settles the foreground receipt when the job is aborted before startup", async () => {
		class PausedStartupEnv extends ScriptedEnv {
			readonly enteredExec = deferred();
			readonly allowStartup = deferred();

			override async exec(
				command: string | readonly string[],
				options: ShellExecOptions | undefined,
				callContext: Context,
			): Promise<Result<ShellExecResult, ExecutionError>> {
				this.enteredExec.resolve();
				await awaitWithContext(this.allowStartup.promise, callContext);
				return super.exec(command, options, callContext);
			}
		}
		const directory = await mkdtemp(join(tmpdir(), "pi-bash-prestart-abort-"));
		const env = new PausedStartupEnv({ cwd: directory });
		const setup = chatSetup();
		setup.registry.install(defineExtension({ name: "bash-policy", tools: [createBashTool()] }));
		setup.faux.setResponses([
			fauxAssistantMessage([fauxToolCall("bash", { command: "printf body", yield_time_ms: 0 })], {
				stopReason: "toolUse",
			}),
			...Array.from({ length: 4 }, () => fauxAssistantMessage([fauxText("done")])),
		]);
		const { harness, root } = await openChat(new MemoryStorage(), setup, { env });
		resources.push({ harness, directory });
		const watchReady = deferred();
		const originalWatchDoc = harness.watchDoc;
		harness.watchDoc = ((...args: readonly unknown[]) => {
			const watching = (originalWatchDoc as (...args: readonly unknown[]) => Promise<unknown>).apply(harness, [
				...args,
			]);
			if (args[0] === CommandDoc)
				watching.then(
					() => watchReady.resolve(),
					(error: unknown) => watchReady.reject(error),
				);
			return watching;
		}) as typeof harness.watchDoc;
		let commandId: TaskId | undefined;
		const unsubscribe = harness.subscribeCommits((publication) => {
			for (const change of publication.changes) {
				if (change.type === "task" && change.value.kind === "pi.bash-command") commandId = change.value.id;
			}
		});
		try {
			await root.submit({ type: "input", content: "run" }, context);
			await env.enteredExec.promise;
			await watchReady.promise;
			if (commandId === undefined) throw new Error("The command job was not admitted");
			await harness.abortTask(commandId, context);
			await waitFor(async () => (await toolResult(root)) !== undefined, 750);
			expect((await toolResult(root))?.isError).toBe(true);
			expect(env.commands).toEqual([]);
		} finally {
			harness.watchDoc = originalWatchDoc;
			unsubscribe();
			env.allowStartup.resolve();
			env.finish.resolve();
		}
	});

	it("does not re-admit explicit-yield Bash after a replay-safe wrapper is reopened", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-bash-safe-replay-"));
		const env = new ScriptedEnv({ cwd: directory });
		const setup = chatSetup();
		const base = createBashTool();
		setup.registry.install(defineExtension({ name: "bash-policy", tools: [{ ...base, replay: "safe" as const }] }));
		const reachedHook = deferred();
		const releaseHook = deferred();
		addHooks(setup.registry, ToolTask, {
			afterTool: async (_call, result, _runtime, callContext) => {
				reachedHook.resolve();
				await awaitWithContext(releaseHook.promise, callContext);
				return result;
			},
		});
		setup.faux.setResponses([
			fauxAssistantMessage([fauxToolCall("bash", { command: "printf body", yield_time_ms: 0 })], {
				stopReason: "toolUse",
			}),
			...Array.from({ length: 8 }, () => fauxAssistantMessage([fauxText("done")])),
		]);
		const storagePath = join(directory, "session.sqlite");
		const { harness, root } = await openChat(await openNodeSqliteStorage(storagePath), setup, { env });
		resources.push({ harness, directory });
		try {
			await root.submit({ type: "input", content: "run" }, context);
			await env.launched.promise;
			await reachedHook.promise;
			await harness.close(context);
			const reopened = await openChat(await openNodeSqliteStorage(storagePath), setup, { env });
			resources.push({ harness: reopened.harness, directory });
			reopened.harness.resume();
			await waitFor(async () => env.commands.length > 1 || (await toolResult(reopened.root)) !== undefined, 2000);
			expect(env.commands).toHaveLength(1);
			expect((await toolResult(reopened.root))?.isError).toBe(true);
		} finally {
			releaseHook.resolve();
			env.finish.resolve();
		}
	});

	it("keeps the phase-pinned Bash factory when it is replaced after reservation but before launch", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-bash-replacement-"));
		const env = new ScriptedEnv({ cwd: directory });
		const setup = chatSetup();
		setup.registry.install(
			defineExtension({
				name: "bash-policy",
				tools: [createBashTool({ commandPrefix: "printf 'original-prefix\\n'" })],
			}),
		);
		setup.faux.setResponses([
			fauxAssistantMessage([fauxToolCall("bash", { command: "printf body", yield_time_ms: 0 })], {
				stopReason: "toolUse",
			}),
			...Array.from({ length: 4 }, () => fauxAssistantMessage([fauxText("done")])),
		]);
		const jobEnteringEnv = deferred();
		const permitLaunch = deferred();
		let envBuilds = 0;
		const { harness, root } = await openChat(new MemoryStorage(), setup, {
			env: async () => {
				// The third build is the command task's: its phase was reserved before runtime.env() runs.
				if (++envBuilds === 3) {
					jobEnteringEnv.resolve();
					await permitLaunch.promise;
				}
				return env;
			},
		});
		resources.push({ harness, directory });
		try {
			await root.submit({ type: "input", content: "run" }, context);
			await jobEnteringEnv.promise;
			expect(env.commands).toEqual([]);
			expect(
				(await harness.inspect(context)).tasks.some(
					({ record }) => record.kind === "pi.bash-command" && record.state.status === "running",
				),
			).toBe(true);
			setup.registry.install(
				defineExtension({
					name: "bash-policy",
					tools: [createBashTool({ commandPrefix: "printf 'replacement-prefix\\n'" })],
				}),
			);
			permitLaunch.resolve();
			await env.launched.promise;
			env.finish.resolve();
			await waitFor(async () => (await toolResult(root)) !== undefined);
			expect(env.commands).toHaveLength(1);
			expect(occurrences(env.commands[0]!, "original-prefix")).toBe(1);
			expect(env.commands[0]).not.toContain("replacement-prefix");
			expect((await toolResult(root))?.isError).toBe(false);
		} finally {
			permitLaunch.resolve();
			env.finish.resolve();
		}
	});

	it("fails closed if the Bash factory changes after admission but before phase reservation", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-bash-pre-reservation-"));
		const env = new ScriptedEnv({ cwd: directory });
		const setup = chatSetup();
		setup.registry.install(defineExtension({ name: "bash-policy", tools: [createBashTool()] }));
		setup.faux.setResponses([
			fauxAssistantMessage([fauxToolCall("bash", { command: "printf body", yield_time_ms: 0 })], {
				stopReason: "toolUse",
			}),
			...Array.from({ length: 4 }, () => fauxAssistantMessage([fauxText("done")])),
		]);
		const { harness, root } = await openChat(new MemoryStorage(), setup, { env });
		resources.push({ harness, directory });
		let replacements = 0;
		// Commit observers run on the Session line before the scheduler can reserve the newly admitted task.
		const unsubscribe = harness.subscribeCommits((publication) => {
			if (
				replacements !== 0 ||
				!publication.changes.some((change) => change.type === "task" && change.value.kind === "pi.bash-command")
			)
				return;
			replacements++;
			setup.registry.install(
				defineExtension({ name: "bash-policy", tools: [createBashTool({ commandPrefix: "printf replaced" })] }),
			);
		});
		try {
			await root.submit({ type: "input", content: "run" }, context);
			await waitFor(async () => (await toolResult(root)) !== undefined);
			expect(replacements).toBe(1);
			expect(env.commands).toEqual([]);
			expect((await toolResult(root))?.isError).toBe(true);
		} finally {
			unsubscribe();
			env.finish.resolve();
		}
	});
});
