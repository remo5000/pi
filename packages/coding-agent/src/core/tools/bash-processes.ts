import { randomUUID } from "node:crypto";
import type { AgentToolResult } from "@earendil-works/pi-agent-core";
import type { BashToolDetails, BashToolOutput } from "./bash.ts";
import { DEFAULT_MAX_BYTES, truncateMiddle, truncateTail } from "./truncate.ts";

type CommandResult = AgentToolResult<BashToolDetails | undefined>;
type Outcome = { ok: true; result: CommandResult } | { ok: false; error: unknown };

export type BashCompletion = {
	command: string;
	process_id: string;
	full_output_path: string;
} & ({ result: Extract<BashToolOutput, { status: "exited" }>; content: CommandResult["content"] } | { error: string });

interface BashExecution {
	run: (signal: AbortSignal) => Promise<CommandResult>;
	snapshot: (processId: string) => Promise<CommandResult>;
	fullOutputPath: () => string;
}

/** Owns one session's yielded commands independently of tool invocations or codemode VMs. */
export class BashProcessManager {
	private readonly pending = new Map<string, { abort: () => void; completion: Promise<void> }>();
	private readonly listeners = new Set<(completion: BashCompletion) => unknown>();
	private disposed = false;

	constructor(onComplete?: (completion: BashCompletion) => unknown) {
		if (onComplete) this.listeners.add(onComplete);
	}

	onCompletion(listener: (completion: BashCompletion) => unknown): () => void {
		if (this.disposed) throw new Error("Bash process manager is disposed");
		this.listeners.add(listener);
		return () => {
			this.listeners.delete(listener);
		};
	}

	/** Cancel current jobs without invalidating tools reused after reload. */
	async abortAll(): Promise<void> {
		const jobs = [...this.pending.values()];
		for (const job of jobs) job.abort();
		await Promise.all(jobs.map(({ completion }) => completion));
	}

	dispose(): void {
		this.disposed = true;
		for (const job of this.pending.values()) job.abort();
		this.listeners.clear();
	}

	async waitForAll(): Promise<void> {
		await Promise.all([...this.pending.values()].map(({ completion }) => completion));
	}

	async execute(
		command: string,
		yieldTimeMs: number,
		execution: BashExecution,
		signal?: AbortSignal,
	): Promise<CommandResult> {
		if (this.disposed) throw new Error("Bash process manager is disposed");
		if (signal?.aborted) throw new Error("Command aborted");
		const processId = randomUUID();
		const controller = new AbortController();
		const onAbort = () => controller.abort();
		signal?.addEventListener("abort", onAbort, { once: true });
		let yielded = false;
		let suppressNotification = false;
		let outcome: Outcome | undefined;
		const completion = execution
			.run(controller.signal)
			.then(
				(result): Outcome => ({ ok: true, result }),
				(error: unknown): Outcome => ({ ok: false, error }),
			)
			.then(async (completed) => {
				outcome = completed;
				if (!yielded || this.disposed || suppressNotification) return;
				const fields = { command, process_id: processId, full_output_path: execution.fullOutputPath() };
				const notification: BashCompletion = completed.ok
					? {
							...fields,
							result: completed.result.structuredContent as Extract<BashToolOutput, { status: "exited" }>,
							content: completed.result.content,
						}
					: {
							...fields,
							error: completed.error instanceof Error ? completed.error.message : String(completed.error),
						};
				await Promise.all(
					[...this.listeners].map(async (listener) => {
						try {
							await listener(notification);
						} catch (error) {
							console.error("bash completion listener failed:", error);
						}
					}),
				);
			})
			.finally(() => this.pending.delete(processId));
		this.pending.set(processId, {
			completion,
			abort: () => {
				suppressNotification = true;
				controller.abort();
			},
		});

		let yieldTimer: NodeJS.Timeout | undefined;
		try {
			await Promise.race([
				completion,
				new Promise<void>((resolve) => {
					yieldTimer = setTimeout(resolve, yieldTimeMs);
				}),
			]);
			const snapshot = outcome ? undefined : await execution.snapshot(processId);
			if (controller.signal.aborted) await completion;
			if (outcome) {
				if (!outcome.ok) throw outcome.error;
				return outcome.result;
			}
			if (!snapshot) throw new Error("Missing running command snapshot");
			// Codemode cancels invocation signals when its VM exits; the session now owns this process.
			yielded = true;
			return snapshot;
		} catch (error) {
			controller.abort();
			await completion;
			throw error;
		} finally {
			if (yieldTimer) clearTimeout(yieldTimer);
			signal?.removeEventListener("abort", onAbort);
		}
	}
}

/** Retrieve process ownership without changing the tool's argument type. */
export function getBashProcessManager(tool: object): BashProcessManager | undefined {
	return "processManager" in tool && tool.processManager instanceof BashProcessManager
		? tool.processManager
		: undefined;
}

export function formatBashCompletion(completion: BashCompletion): string {
	const status = "error" in completion ? "Command failed" : `Command exited with code ${completion.result.exit_code}`;
	const output =
		"error" in completion
			? completion.error
			: completion.content
					.filter((part) => part.type === "text")
					.map((part) => part.text)
					.join("\n");
	const header = `[BASH ${completion.process_id}] ${status}`;
	const command = truncateMiddle(completion.command, 1024).content;
	const footer = `\nCommand: ${command}\nFull output: ${completion.full_output_path}`;
	// Inline scripts and error messages need the same bound as stdout; preserve the terminal status and log path.
	const maxBytes = DEFAULT_MAX_BYTES - Buffer.byteLength(`${header}\n${footer}`);
	if (maxBytes <= 0) throw new Error("Bash completion metadata exceeds the output limit");
	return `${header}\n${truncateTail(output || "(no output)", { maxBytes }).content}${footer}`;
}
