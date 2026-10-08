import { constants } from "node:fs";
import { access as fsAccess } from "node:fs/promises";
import { constants as osConstants } from "node:os";
import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { spawn } from "child_process";
import { type Static, Type } from "typebox";
import { waitForChildProcess } from "../../utils/child-process.ts";
import {
	getShellConfig,
	getShellEnv,
	killProcessTree,
	type ShellConfig,
	trackDetachedChildPid,
	untrackDetachedChildPid,
} from "../../utils/shell.ts";
import type { ExtensionContext, ToolDefinition } from "../extensions/types.ts";
import { BashProcessManager, getBashProcessManager } from "./bash-processes.ts";
import { OutputAccumulator } from "./output-accumulator.ts";
import { BASH_UPDATE_THROTTLE_MS, createShellRenderers } from "./renderers/bash.ts";
import { wrapToolDefinition } from "./tool-definition-wrapper.ts";
import { DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, formatSize, type TruncationResult } from "./truncate.ts";

const MAX_TIMEOUT_MS = 2_147_483_647;
/** Output limit of `structuredContent.output`, which programmatic callers such as codemode scripts receive. */
const STRUCTURED_OUTPUT_MAX_BYTES = 1024 * 1024;
const MAX_TIMEOUT_SECONDS = MAX_TIMEOUT_MS / 1000;

function resolveTimeoutMs(timeout: number | undefined): number | undefined {
	if (timeout === undefined) return undefined;
	if (!Number.isFinite(timeout) || timeout <= 0) {
		throw new Error("Invalid timeout: must be a finite number of seconds");
	}

	const timeoutMs = timeout * 1000;
	if (timeoutMs > MAX_TIMEOUT_MS) {
		throw new Error(`Invalid timeout: maximum is ${MAX_TIMEOUT_SECONDS} seconds`);
	}
	return timeoutMs;
}

const shellSchema = Type.Object({
	command: Type.String({ description: "Shell command to execute" }),
	timeout: Type.Optional(Type.Number({ description: "Timeout in seconds (optional, no default timeout)" })),
});
const bashSchema = Type.Object({
	...shellSchema.properties,
	yield_time_ms: Type.Optional(
		Type.Integer({
			minimum: 0,
			maximum: 30_000,
			description: "Wait before yielding in milliseconds; omitted waits for completion, 0 yields immediately",
		}),
	),
});

export const bashToolSystemPromptContribution = {
	snippet: "Execute bash commands (ls, grep, find, etc.)",
	guidelines: ["You can inspect PI_* environment variables for current model and session details."],
} as const;

export type BashToolInput = Static<typeof bashSchema>;
export type ShellToolInput = Static<typeof shellSchema>;

/**
 * Result for programmatic callers such as codemode scripts. A non-zero exit code is an error result for the model, but scripts still resolve to this value.
 * `output` is not limited like the model-facing output: callers decide how much of it reaches the model.
 */
const outputFields = {
	output: Type.String({ description: "Combined stdout and stderr, possibly truncated" }),
	truncated: Type.Boolean(),
	full_output_path: Type.Optional(Type.String({ description: "Full output, when truncated or running" })),
	wall_time_seconds: Type.Number(),
};
const shellOutputSchema = Type.Object({ ...outputFields, exit_code: Type.Number() });
const bashOutputSchema = Type.Union([
	Type.Object({ status: Type.Literal("exited"), ...outputFields, exit_code: Type.Number() }),
	Type.Object({
		status: Type.Literal("running"),
		...outputFields,
		process_id: Type.String(),
		full_output_path: Type.String(),
	}),
]);

export type BashToolOutput = Static<typeof bashOutputSchema>;

export interface BashToolDetails {
	truncation?: TruncationResult;
	fullOutputPath?: string;
}

/**
 * Pluggable operations for the bash tool.
 * Override these to delegate command execution to remote systems (for example SSH).
 */
export interface BashOperations {
	/**
	 * Execute a command and stream output.
	 * @param command The command to execute
	 * @param cwd Working directory
	 * @param options Execution options
	 * @returns Promise resolving to the exit code. Report signal terminations as 128 + signal number;
	 * a null exit code is treated as a failed command.
	 */
	exec: (
		command: string,
		cwd: string,
		options: {
			/** Return a promise to pause output until the consumer finishes. Other return values are ignored. */
			onData: (data: Buffer) => unknown;
			signal?: AbortSignal;
			timeout?: number;
			env?: NodeJS.ProcessEnv;
		},
	) => Promise<{ exitCode: number | null }>;
}

/** Shared process execution used by the built-in shell tools. */
export function createLocalShellOperations(shellName: string, resolveShellConfig: () => ShellConfig): BashOperations {
	return {
		exec: async (command, cwd, { onData, signal, timeout, env }) => {
			const timeoutMs = resolveTimeoutMs(timeout);
			if (signal?.aborted) {
				throw new Error("aborted");
			}
			const shellConfig = resolveShellConfig();
			try {
				await fsAccess(cwd, constants.F_OK);
			} catch {
				throw new Error(`Working directory does not exist: ${cwd}\nCannot execute ${shellName} commands.`);
			}

			const commandFromStdin = shellConfig.commandTransport === "stdin";
			const child = spawn(shellConfig.shell, commandFromStdin ? shellConfig.args : [...shellConfig.args, command], {
				cwd,
				detached: process.platform !== "win32",
				env: env ?? getShellEnv(),
				stdio: [commandFromStdin ? "pipe" : "ignore", "pipe", "pipe"],
				windowsHide: true,
			});
			if (commandFromStdin) {
				child.stdin?.on("error", () => {});
				child.stdin?.end(command);
			}
			if (child.pid) trackDetachedChildPid(child.pid);
			let timedOut = false;
			let timeoutHandle: NodeJS.Timeout | undefined;
			let outputError: Error | undefined;
			const pendingOutput = new Set<Promise<void>>();
			const handleData = (data: Buffer) => {
				if (outputError) return;
				try {
					const consumed = onData(data);
					if (!(consumed instanceof Promise)) return;
					child.stdout?.pause();
					child.stderr?.pause();
					const pending = consumed
						.catch((error: unknown) => {
							outputError = error instanceof Error ? error : new Error(String(error));
							if (child.pid) killProcessTree(child.pid);
						})
						.finally(() => {
							pendingOutput.delete(pending);
							if (pendingOutput.size === 0) {
								child.stdout?.resume();
								child.stderr?.resume();
							}
						});
					pendingOutput.add(pending);
				} catch (error) {
					outputError = error instanceof Error ? error : new Error(String(error));
					if (child.pid) killProcessTree(child.pid);
				}
			};
			const onAbort = () => {
				if (child.pid) killProcessTree(child.pid);
			};

			try {
				// Set timeout if provided.
				if (timeoutMs !== undefined) {
					timeoutHandle = setTimeout(() => {
						timedOut = true;
						if (child.pid) killProcessTree(child.pid);
					}, timeoutMs);
				}
				// Stream stdout and stderr.
				child.stdout?.on("data", handleData);
				child.stderr?.on("data", handleData);
				// Handle abort signal by killing the entire process tree.
				if (signal) {
					if (signal.aborted) onAbort();
					else signal.addEventListener("abort", onAbort, { once: true });
				}
				// Handle shell spawn errors and wait for the process to terminate without hanging
				// on inherited stdio handles held by detached descendants.
				const exitCode = await waitForChildProcess(child, () => pendingOutput.size > 0);
				await Promise.all(pendingOutput);
				if (outputError) throw outputError;
				if (signal?.aborted) {
					throw new Error("aborted");
				}
				if (timedOut) {
					throw new Error(`timeout:${timeout}`);
				}
				// A signal-killed shell has no exit code. Use the standard shell convention so
				// callers do not mistake the termination for a successful command.
				const signalCode = child.signalCode;
				return { exitCode: exitCode ?? (signalCode ? 128 + (osConstants.signals[signalCode] ?? 0) : 1) };
			} finally {
				if (child.pid) untrackDetachedChildPid(child.pid);
				if (timeoutHandle) clearTimeout(timeoutHandle);
				if (signal) signal.removeEventListener("abort", onAbort);
			}
		},
	};
}

/**
 * Create bash operations using pi's built-in local shell execution backend.
 *
 * This is useful for extensions that intercept user_bash and still want pi's
 * standard local shell behavior while wrapping or rewriting commands.
 */
export function createLocalBashOperations(options?: { shellPath?: string }): BashOperations {
	return createLocalShellOperations("bash", () => getShellConfig(options?.shellPath));
}

export interface BashSpawnContext {
	command: string;
	cwd: string;
	env: NodeJS.ProcessEnv;
}

export type BashSpawnHook = (context: BashSpawnContext) => BashSpawnContext;

export function resolveSpawnContext(
	command: string,
	cwd: string,
	spawnHook: BashSpawnHook | undefined,
	exposeSessionEnvironment: boolean,
	ctx: ExtensionContext | undefined,
): BashSpawnContext {
	const env = { ...getShellEnv() };
	delete env.PI_SESSION_ID;
	delete env.PI_SESSION_FILE;
	delete env.PI_PROVIDER;
	delete env.PI_MODEL;
	delete env.PI_REASONING_LEVEL;
	if (exposeSessionEnvironment && ctx) {
		const model = ctx.model;
		env.PI_SESSION_ID = ctx.sessionManager.getSessionId();
		const sessionFile = ctx.sessionManager.getSessionFile();
		if (sessionFile) env.PI_SESSION_FILE = sessionFile;
		if (model) {
			env.PI_PROVIDER = model.provider;
			env.PI_MODEL = model.id;
		}
		if (ctx.thinkingLevel) env.PI_REASONING_LEVEL = ctx.thinkingLevel;
	}
	const baseContext: BashSpawnContext = { command, cwd, env };
	return spawnHook ? spawnHook(baseContext) : baseContext;
}

export interface BashToolOptions {
	/** Custom operations for command execution. Default: local shell */
	operations?: BashOperations;
	/** Command prefix prepended to every command (for example shell setup commands) */
	commandPrefix?: string;
	/** Optional explicit shell path from settings */
	shellPath?: string;
	/** Expose current Pi session metadata as PI_* environment variables. Default: true */
	exposeSessionEnvironment?: boolean;
	/** Hook to adjust command, cwd, or env before execution */
	spawnHook?: BashSpawnHook;
	/** Use one manager per session; dispose standalone owners when done. */
	processManager?: BashProcessManager;
}

export type BashRenderState = {
	startedAt: number | undefined;
	endedAt: number | undefined;
	interval: NodeJS.Timeout | undefined;
};

export interface ShellToolConfig {
	name: string;
	label: string;
	shellName: string;
	prompt: string;
	promptSnippet: string;
	promptGuidelines?: readonly string[];
	tempFilePrefix: string;
}

export function createShellToolDefinition(
	cwd: string,
	config: ShellToolConfig,
	options?: BashToolOptions,
	processManager?: BashProcessManager,
): ToolDefinition<typeof shellSchema, BashToolDetails | undefined, BashRenderState> {
	const ops = options?.operations ?? createLocalBashOperations({ shellPath: options?.shellPath });
	const commandPrefix = options?.commandPrefix;
	const exposeSessionEnvironment = options?.exposeSessionEnvironment ?? true;
	const spawnHook = options?.spawnHook;
	return {
		name: config.name,
		label: config.label,
		description: `Execute a ${config.shellName} command in the current working directory. Returns stdout and stderr. Output is truncated to last ${DEFAULT_MAX_LINES} lines or ${DEFAULT_MAX_BYTES / 1024}KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.`,
		promptSnippet: config.promptSnippet,
		promptGuidelines: exposeSessionEnvironment && config.promptGuidelines ? [...config.promptGuidelines] : undefined,
		parameters: shellSchema,
		outputSchema: processManager ? bashOutputSchema : shellOutputSchema,
		constrainedSampling: { type: "json_schema", strict: "prefer" },
		async execute(
			_toolCallId,
			{ command, timeout, yield_time_ms }: BashToolInput,
			signal?: AbortSignal,
			onUpdate?,
			ctx?: ExtensionContext,
		) {
			if (signal?.aborted) throw new Error("Command aborted");
			if (
				yield_time_ms !== undefined &&
				(!Number.isInteger(yield_time_ms) || yield_time_ms < 0 || yield_time_ms > 30_000)
			) {
				throw new Error("Invalid yield_time_ms: must be an integer between 0 and 30000");
			}
			const resolvedCommand = commandPrefix ? `${commandPrefix}\n${command}` : command;
			const spawnContext = resolveSpawnContext(
				resolvedCommand,
				ctx?.cwd || cwd,
				spawnHook,
				exposeSessionEnvironment,
				ctx,
			);
			const output = new OutputAccumulator({ tempFilePrefix: config.tempFilePrefix });
			let acceptingOutput = true;
			let acceptingUpdates = true;
			let updateTimer: NodeJS.Timeout | undefined;
			let updateDirty = false;
			let lastUpdateAt = 0;

			const emitOutputUpdate = () => {
				if (!acceptingUpdates || !onUpdate || !updateDirty) return;
				updateDirty = false;
				lastUpdateAt = Date.now();
				const snapshot = output.snapshot({ persistIfTruncated: true });
				onUpdate({
					content: [{ type: "text", text: snapshot.content || "" }],
					details: {
						truncation: snapshot.truncation.truncated ? snapshot.truncation : undefined,
						fullOutputPath: snapshot.fullOutputPath,
					},
				});
			};

			const clearUpdateTimer = () => {
				if (updateTimer) {
					clearTimeout(updateTimer);
					updateTimer = undefined;
				}
			};

			const scheduleOutputUpdate = () => {
				if (!acceptingUpdates || !onUpdate) return;
				updateDirty = true;
				const delay = BASH_UPDATE_THROTTLE_MS - (Date.now() - lastUpdateAt);
				if (delay <= 0) {
					clearUpdateTimer();
					emitOutputUpdate();
					return;
				}
				updateTimer ??= setTimeout(() => {
					updateTimer = undefined;
					emitOutputUpdate();
				}, delay);
			};

			if (onUpdate) {
				onUpdate({ content: [], details: undefined });
			}

			const handleData = (data: Buffer) => {
				if (!acceptingOutput) return;
				output.append(data);
				scheduleOutputUpdate();
				return output.flushTempFile();
			};

			const finishOutput = async () => {
				acceptingOutput = false;
				output.finish();
				clearUpdateTimer();
				emitOutputUpdate();
				const snapshot = output.snapshot({ persistIfTruncated: true });
				await output.closeTempFile();
				return snapshot;
			};

			const formatOutput = (snapshot: Awaited<ReturnType<typeof finishOutput>>, emptyText = "(no output)") => {
				const truncation = snapshot.truncation;
				let text = snapshot.content || emptyText;
				let details: BashToolDetails | undefined;
				if (truncation.truncated) {
					details = { truncation, fullOutputPath: snapshot.fullOutputPath };
					const startLine = truncation.totalLines - truncation.outputLines + 1;
					const endLine = truncation.totalLines;
					if (truncation.lastLinePartial) {
						const lastLineSize = formatSize(output.getLastLineBytes());
						text += `\n\n[Showing last ${formatSize(truncation.outputBytes)} of line ${endLine} (line is ${lastLineSize}). Full output: ${snapshot.fullOutputPath}]`;
					} else if (truncation.truncatedBy === "lines") {
						text += `\n\n[Showing lines ${startLine}-${endLine} of ${truncation.totalLines}. Full output: ${snapshot.fullOutputPath}]`;
					} else {
						text += `\n\n[Showing lines ${startLine}-${endLine} of ${truncation.totalLines} (${formatSize(DEFAULT_MAX_BYTES)} limit). Full output: ${snapshot.fullOutputPath}]`;
					}
				}
				return { text, details };
			};

			const appendStatus = (text: string, status: string) => `${text ? `${text}\n\n` : ""}${status}`;
			const startedAt = performance.now();
			const run = async (processSignal = signal): Promise<AgentToolResult<BashToolDetails | undefined>> => {
				if (yield_time_ms !== undefined) output.persistFullOutput();

				try {
					let exitCode: number | null;
					try {
						const result = await ops.exec(spawnContext.command, spawnContext.cwd, {
							onData: handleData,
							signal: processSignal,
							timeout,
							env: spawnContext.env,
						});
						exitCode = result.exitCode;
					} catch (err) {
						const snapshot = await finishOutput();
						const { text } = formatOutput(snapshot, "");
						if (err instanceof Error && err.message === "aborted") {
							throw new Error(appendStatus(text, "Command aborted"));
						}
						if (err instanceof Error && err.message.startsWith("timeout:")) {
							const timeoutSecs = err.message.split(":")[1];
							throw new Error(appendStatus(text, `Command timed out after ${timeoutSecs} seconds`));
						}
						throw err;
					}

					const snapshot = await finishOutput();
					const { text: outputText, details } = formatOutput(snapshot);
					if (exitCode === null) {
						throw new Error(appendStatus(outputText, "Command terminated without an exit code"));
					}
					const wallTimeSeconds = Math.round((performance.now() - startedAt) / 100) / 10;
					const fullOutput = await output.readFullOutput(STRUCTURED_OUTPUT_MAX_BYTES);
					const structuredContent = {
						...(processManager ? { status: "exited" as const } : {}),
						output: fullOutput.content,
						truncated: fullOutput.truncated,
						...(fullOutput.truncated && snapshot.fullOutputPath
							? { full_output_path: snapshot.fullOutputPath }
							: {}),
						exit_code: exitCode,
						wall_time_seconds: wallTimeSeconds,
					};
					if (exitCode !== 0) {
						return {
							content: [
								{ type: "text", text: appendStatus(outputText, `Command exited with code ${exitCode}`) },
							],
							details,
							structuredContent,
							isError: true,
						};
					}
					return { content: [{ type: "text", text: outputText }], details, structuredContent };
				} finally {
					clearUpdateTimer();
				}
			};
			if (!processManager || yield_time_ms === undefined) return run();
			try {
				return await processManager.execute(
					command,
					yield_time_ms,
					{
						run,
						fullOutputPath: () => output.persistFullOutput(),
						snapshot: async (processId) => {
							await output.flushTempFile();
							const snapshot = output.snapshot({ persistIfTruncated: true });
							const { text, details } = formatOutput(snapshot);
							const fullOutputPath = output.persistFullOutput();
							const structuredContent: Extract<BashToolOutput, { status: "running" }> = {
								status: "running",
								process_id: processId,
								output: snapshot.content,
								truncated: snapshot.truncation.truncated,
								full_output_path: fullOutputPath,
								wall_time_seconds: Math.round((performance.now() - startedAt) / 100) / 10,
							};
							return {
								content: [
									{
										type: "text",
										text: appendStatus(
											text,
											`Command still running (process ${processId}). Completion will be reported later.\nFull output: ${fullOutputPath}`,
										),
									},
								],
								details: { ...details, fullOutputPath },
								structuredContent,
							};
						},
					},
					signal,
				);
			} finally {
				acceptingUpdates = false;
				clearUpdateTimer();
			}
		},
		...createShellRenderers(config.prompt),
	};
}

const bashToolConfig: ShellToolConfig = {
	name: "bash",
	label: "bash",
	shellName: "bash",
	prompt: "$",
	promptSnippet: bashToolSystemPromptContribution.snippet,
	promptGuidelines: bashToolSystemPromptContribution.guidelines,
	tempFilePrefix: "pi-bash",
};

export function createBashToolDefinition(
	cwd: string,
	options?: BashToolOptions,
): ToolDefinition<typeof bashSchema, BashToolDetails | undefined, BashRenderState> {
	const processManager = options?.processManager ?? new BashProcessManager();
	const shellDefinition = createShellToolDefinition(cwd, bashToolConfig, options, processManager);
	const definition: ToolDefinition<typeof bashSchema, BashToolDetails | undefined, BashRenderState> = {
		...shellDefinition,
		parameters: bashSchema,
		description: `${shellDefinition.description} Waits for completion unless yield_time_ms is supplied; yielding keeps the process running and reports completion later.`,
		promptGuidelines: [
			...(shellDefinition.promptGuidelines ?? []),
			"With yield_time_ms, a running result is not success. Wait for the completion notification before dependent work; read full_output_path for later output.",
		],
	};
	Object.assign(definition, { processManager });
	return definition;
}

export function createBashTool(cwd: string, options?: BashToolOptions): AgentTool<typeof bashSchema> {
	const definition = createBashToolDefinition(cwd, options);
	const tool = wrapToolDefinition(definition);
	const processManager = getBashProcessManager(definition);
	if (!processManager) throw new Error("Bash tool definition has no process manager");
	Object.assign(tool, {
		processManager,
		promptSnippet: definition.promptSnippet,
		promptGuidelines: definition.promptGuidelines,
	});
	return tool;
}
