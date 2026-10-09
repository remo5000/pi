/** A bounded snapshot; full output stays in the execution environment's file namespace. */
export type CommandOutput = {
	output: string;
	truncated: boolean;
	wall_time_seconds: number;
	full_output_path?: string;
};

export type CommandResult =
	| (CommandOutput & { status: "exited"; exit_code: number })
	| (CommandOutput & {
			status: "running";
			/** The execution backend's PID, not a transport/client PID. Backends may omit it. */
			pid?: number;
			full_output_path: string;
	  });

/** Registry states also represent startup and execution failures; neither implies a successful exit. */
export type CommandState =
	| CommandResult
	| (CommandOutput & { status: "starting"; full_output_path: string })
	| (CommandOutput & { status: "failed"; error: string; pid?: number });
