import type { Context, JsonValue } from "@earendil-works/chord";
import type { ExecutionEnv } from "../env/index.ts";
import type { Task, TaskId, TaskOptions, TaskRuntime } from "../types.ts";
import type { ToolExecutionApi } from "./types.ts";

/** The same invocation-bound capabilities for a foreground tool and a background command. */
export function createToolExecutionApi<I, S, R, H extends object, D extends JsonValue>(
	runtime: TaskRuntime<I, S, R, H>,
	options: Pick<ToolExecutionApi<D>, "callId" | "output" | "outputWindow" | "diagnostic" | "details">,
	env: ExecutionEnv | undefined,
): ToolExecutionApi<D> {
	return {
		...options,
		env,
		taskId: runtime.taskId,
		conversationId: runtime.conversationId,
		registry: runtime.registry,
		agent: runtime.agent,
		models: runtime.models,
		commit: async (change, context) => {
			let result: Awaited<ReturnType<typeof change>> | undefined;
			await runtime.commit(async (tx) => {
				result = await change(tx);
				return undefined;
			}, context);
			return result as Awaited<ReturnType<typeof change>>;
		},
		memo: runtime.memo,
		createTask: async <TI, TS extends { phase: string }, TR, TH extends object>(
			task: Task<TI, TS, TR, TH>,
			input: TI,
			options: Omit<TaskOptions, "conversationId">,
			context: Context,
		): Promise<TaskId<TR>> => {
			let id: TaskId<TR> | undefined;
			await runtime.commit(async (tx) => {
				id = await tx.createTask(task, input, options);
				return undefined;
			}, context);
			return id!;
		},
		getTask: runtime.getTask,
		waitForTask: runtime.waitForTask,
		conversation: runtime.conversation,
		snapshot: runtime.snapshot,
		snapshotAsOf: runtime.snapshotAsOf,
		watchDoc: runtime.watchDoc,
	};
}
