import type { Context } from "@earendil-works/chord";
import { uuidv7 } from "@earendil-works/pi-ai/utils/uuid";
import { type Static, Type } from "typebox";
import type { CommandResult } from "../env/command-state.ts";
import { defineTool } from "../harness/define.ts";
import { OutputBuffer } from "../harness/output.ts";
import type { ToolExecutionApi, ToolRegistration } from "../harness/types.ts";
import { DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES } from "../truncate.ts";
import {
	BashCommandTask,
	commandFactory,
	type ManagedCommandApi,
	managedCommand,
	yieldBashCommand,
} from "./bash-command.ts";
import { requireEnv } from "./env.ts";

const MAX_TIMEOUT_SECONDS = 2_147_483_647 / 1000;

const bashSchema = Type.Object({
	command: Type.String({ description: "Bash command to execute" }),
	timeout: Type.Optional(Type.Number({ description: "Timeout in seconds (optional, no default timeout)" })),
	yield_time_ms: Type.Optional(
		Type.Integer({
			minimum: 0,
			maximum: 30000,
			description:
				"Allow returning while the command is running after this many milliseconds. Omit to wait for completion; 0 waits only for startup.",
		}),
	),
});

const powershellSchema = Type.Object({
	command: Type.String({ description: "PowerShell command to execute" }),
	timeout: Type.Optional(Type.Number({ description: "Timeout in seconds (optional, no default timeout)" })),
});

export type BashToolInput = Static<typeof bashSchema>;
export type PowerShellToolInput = Static<typeof powershellSchema>;

/** A command about to run, which `prepare` may change: the script, its working directory and environment. */
export interface BashExecution {
	command: string;
	cwd: string;
	env: Record<string, string>;
	inheritEnv: boolean;
}

export type BashPrepare = (execution: BashExecution, api: ToolExecutionApi, context: Context) => void | Promise<void>;

export interface BashToolOptions {
	commandPrefix?: string;
	prepare?: BashPrepare;
}

export interface PowerShellToolOptions {
	/** Lines run before each command. */
	commandPrefix?: string;
	prepare?: BashPrepare;
	/** PowerShell programs to try in order; default `pwsh`, then `powershell`. A program that cannot start is skipped. */
	programs?: readonly string[];
}

function validateYieldTime(value: unknown): void {
	if (value !== undefined && (typeof value !== "number" || !Number.isInteger(value) || value < 0 || value > 30000)) {
		throw new Error("Invalid yield_time_ms: expected an integer from 0 to 30000");
	}
}

function validateTimeout(timeout: number | undefined): void {
	if (timeout === undefined) return;
	if (!Number.isFinite(timeout) || timeout <= 0) {
		throw new Error("Invalid timeout: must be a finite number of seconds");
	}
	if (timeout > MAX_TIMEOUT_SECONDS) {
		throw new Error(`Invalid timeout: maximum is ${MAX_TIMEOUT_SECONDS} seconds`);
	}
}

/** The execution of one call: the command with its prefix, in the environment's working directory, then `prepare`. */
async function prepareExecution(
	command: string,
	options: { commandPrefix?: string; prepare?: BashPrepare } | undefined,
	api: ToolExecutionApi,
	context: Context,
	cwd?: string,
): Promise<BashExecution> {
	const execution: BashExecution = {
		command: options?.commandPrefix ? `${options.commandPrefix}\n${command}` : command,
		cwd: cwd ?? requireEnv(api).cwd,
		env: {},
		inheritEnv: true,
	};
	await options?.prepare?.(execution, api, context);
	return execution;
}

/**
 * Run each command form in turn until one starts, streaming output to `api.output()` within the retained window, and
 * turn the result into the tool's outcome: a spill diagnostic, and a thrown error for a failure or a nonzero exit.
 */
async function runCommand(
	commands: readonly (string | readonly string[])[],
	execution: BashExecution,
	timeout: number | undefined,
	api: ToolExecutionApi,
	context: Context,
	capture?: { output: OutputBuffer; fullOutputPath?: string; onSpawn?: (pid?: number) => Promise<void> },
) {
	const env = requireEnv(api);
	let result: Awaited<ReturnType<typeof env.exec>> | undefined;
	for (const command of commands) {
		result = await env.exec(
			command,
			{
				cwd: execution.cwd,
				env: execution.env,
				inheritEnv: execution.inheritEnv,
				...(timeout === undefined ? {} : { timeout }),
				onOutput: (text, _context, info) => {
					capture?.output.push(text, info.skipped);
					api.output(text, info.skipped);
				},
				...(capture?.onSpawn === undefined ? {} : { onSpawn: capture.onSpawn }),
				spill: {
					afterBytes: DEFAULT_MAX_BYTES,
					afterLines: DEFAULT_MAX_LINES,
					...(capture?.fullOutputPath === undefined ? {} : { path: capture.fullOutputPath }),
				},
				// An environment may then omit output outside the retained tail and report the omission.
				...(api.outputWindow === undefined ? {} : { window: api.outputWindow }),
			},
			context,
		);
		// A program that could not start produced no output; the next form may.
		if (result.ok || result.error.code !== "spawn_error") break;
	}
	if (result === undefined) throw new Error("No command to run");
	const spillPath = result.ok ? result.value.spillPath : result.error.spillPath;
	if (spillPath !== undefined) {
		api.diagnostic({ severity: "info", code: "full_output", message: `Full output: ${spillPath}` });
	}
	if (!result.ok) {
		if (result.error.code === "aborted" && context.abortSignal?.aborted) throw result.error;
		if (result.error.code === "timeout") throw new Error(`Command timed out after ${timeout} seconds`);
		if (result.error.code === "aborted") throw new Error("Command aborted");
		throw result.error;
	}
	if (capture === undefined && result.value.exitCode !== 0)
		throw new Error(`Command exited with code ${result.value.exitCode}`);
	return result.value;
}

/**
 * Runs a command through the environment's shell. Its output streams to `api.output()`, where the Harness keeps the
 * tail within the default limits; the result content is that retained output. The retained window goes to the
 * environment, which may omit output outside it and report how much it omitted, so dropped counts stay exact. Output
 * beyond the limits is spilled to a file whose path is reported as a diagnostic. A nonzero exit or timeout throws, which
 * makes an error result that still carries the output and diagnostics.
 */
export function createBashTool(options?: BashToolOptions): ToolRegistration<typeof bashSchema, CommandResult> {
	const tool: ToolRegistration<typeof bashSchema, CommandResult> & { [commandFactory]: string } = Object.assign(
		defineTool<typeof bashSchema, CommandResult>({
			name: "bash",
			description: `Execute a bash command in the current working directory. Returns combined stdout and stderr. Output is truncated to last ${DEFAULT_MAX_LINES} lines or ${DEFAULT_MAX_BYTES / 1024}KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds. Omit yield_time_ms to wait for completion; set 0..30000 to allow returning running. Running is not success and has no exit code; completion arrives later.`,
			parameters: bashSchema,
			prepareArguments: (args) => {
				// TypeBox would otherwise coerce fractional/null waits into valid integers.
				if (typeof args === "object" && args !== null && "yield_time_ms" in args)
					validateYieldTime(args.yield_time_ms);
				return args as BashToolInput;
			},
			outputLimits: { retain: "tail" },
			async execute(args, api, context) {
				validateTimeout(args.timeout);
				const managed = (api as ManagedCommandApi)[managedCommand];
				validateYieldTime(args.yield_time_ms);
				if (args.yield_time_ms !== undefined && managed === undefined) {
					if ((await api.getTask(api.taskId, context))?.kind === BashCommandTask.definition.name)
						throw new Error("The Bash wrapper did not preserve the managed invocation API");
					return yieldBashCommand({ ...args, yield_time_ms: args.yield_time_ms }, api, context, tool);
				}
				const startedAt = performance.now();
				const output = new OutputBuffer({
					maxBytes: DEFAULT_MAX_BYTES,
					maxLines: DEFAULT_MAX_LINES,
					retain: "tail",
				});
				const execution = await prepareExecution(args.command, options, api, context, managed?.cwd);
				const result = await runCommand([execution.command], execution, args.timeout, api, context, {
					output,
					...(managed === undefined ? {} : { fullOutputPath: managed.fullOutputPath, onSpawn: managed.onSpawn }),
				});
				output.end();
				const snapshot = output.snapshot();
				const details: Extract<CommandResult, { status: "exited" }> = {
					status: "exited",
					exit_code: result.exitCode,
					output: snapshot.text,
					truncated: snapshot.droppedBytes > 0,
					wall_time_seconds: (performance.now() - startedAt) / 1000,
					...(result.spillPath === undefined ? {} : { full_output_path: result.spillPath }),
				};
				if (managed !== undefined) return { details, ...(details.exit_code === 0 ? {} : { isError: true }) };
				if (details.exit_code !== 0) {
					await api.details(details, context);
					throw new Error(`Command exited with code ${details.exit_code}`);
				}
				return { details };
			},
		}),
		{ [commandFactory]: uuidv7() },
	);
	return tool;
}

/** Output in UTF-8 whatever the console's code page, as the coding agent's `powershell` tool does. */
const UTF8_OUTPUT = "try { [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 } catch {}";
const POWERSHELL_ARGS = ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command"] as const;

/**
 * Runs a PowerShell command, like `bash` but through PowerShell instead of the environment's shell: `pwsh` (PowerShell
 * 7), else Windows PowerShell, started directly with the command as an argument, so no other shell parses it.
 */
export function createPowerShellTool(options?: PowerShellToolOptions): ToolRegistration<typeof powershellSchema> {
	const programs = options?.programs ?? ["pwsh", "powershell"];
	return defineTool({
		name: "powershell",
		description: `Execute a PowerShell command in the current working directory. Returns combined stdout and stderr. Output is truncated to last ${DEFAULT_MAX_LINES} lines or ${DEFAULT_MAX_BYTES / 1024}KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.`,
		parameters: powershellSchema,
		outputLimits: { retain: "tail" },
		async execute(args, api, context) {
			validateTimeout(args.timeout);
			const execution = await prepareExecution(args.command, options, api, context);
			const script = `${UTF8_OUTPUT}\n${execution.command}`;
			const commands = programs.map((program) => [program, ...POWERSHELL_ARGS, script]);
			await runCommand(commands, execution, args.timeout, api, context);
			return {};
		},
	});
}
