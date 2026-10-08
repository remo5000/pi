import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ToolDefinition } from "../../src/core/extensions/types.ts";
import {
	BashProcessManager,
	type BashToolOptions,
	createBashTool,
	createBashToolDefinition,
	getBashProcessManager,
} from "../../src/core/tools/index.ts";
import { wrapToolDefinition } from "../../src/core/tools/tool-definition-wrapper.ts";
import { createCodemodeExtension } from "../../src/extensions/codemode/index.ts";
import { createHarness, getAssistantTexts, getMessageText, getToolResult, type Harness } from "./harness.ts";

describe("AgentSession bash yield integration", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) harnesses.pop()?.cleanup();
	});

	it("keeps bash active by default without registering exec_command", async () => {
		const harness = await createHarness();
		harnesses.push(harness);

		expect(harness.session.getActiveToolNames()).toEqual(["read", "bash", "edit", "write"]);
		expect(harness.session.getAllTools().map((tool) => tool.name)).not.toContain("exec_command");
		expect(harness.session.systemPrompt).toContain("- bash:");
		expect(harness.session.systemPrompt).not.toContain("- exec_command:");
	});

	it.skipIf(process.platform === "win32")(
		"kills yielded jobs on bare SDK dispose without binding extensions",
		async () => {
			const harness = await createHarness();
			harnesses.push(harness);
			const releasePath = join(harness.tempDir, "release-after-dispose");
			const code =
				"const fs = require('node:fs'); setInterval(() => { if (fs.existsSync(process.argv[1])) { console.log('should-not-finish'); process.exit(0); } }, 10);";
			const command = `${JSON.stringify(process.execPath)} -e ${JSON.stringify(code)} ${JSON.stringify(releasePath)}`;
			harness.setResponses([
				fauxAssistantMessage([fauxToolCall("bash", { command, yield_time_ms: 0 })], {
					stopReason: "toolUse",
				}),
				fauxAssistantMessage("job started"),
			]);

			await harness.session.prompt("start job");
			const result = getToolResult(harness, "bash");
			expect(getMessageText(result)).toContain("Command still running");
			const logPath = (result.details as { fullOutputPath?: string } | undefined)?.fullOutputPath;
			if (!logPath) throw new Error("No command log path");
			harness.session.dispose();
			writeFileSync(releasePath, "go");
			await new Promise((resolve) => setTimeout(resolve, 150));

			expect(readFileSync(logPath, "utf8")).not.toContain("should-not-finish");
			expect(
				harness.session.messages.some(
					(message) => message.role === "custom" && message.customType === "bash_completion",
				),
			).toBe(false);
		},
	);

	it("disposes a standalone bash manager passed as a session override", async () => {
		let aborted = false;
		const tool = createBashTool(process.cwd(), {
			operations: {
				exec: async (_command, _cwd, { signal }) =>
					new Promise<{ exitCode: number | null }>((_resolve, reject) => {
						signal?.addEventListener(
							"abort",
							() => {
								aborted = true;
								reject(new Error("aborted"));
							},
							{ once: true },
						);
					}),
			},
		});
		const manager = getBashProcessManager(tool);
		if (!manager) throw new Error("Bash tool has no process manager");
		const harness = await createHarness({ tools: [tool] });
		harnesses.push(harness);
		harness.setResponses([
			fauxAssistantMessage([fauxToolCall("bash", { command: "waiting", yield_time_ms: 0 })], {
				stopReason: "toolUse",
			}),
			fauxAssistantMessage("job started"),
		]);

		await harness.session.prompt("start job");
		expect(getMessageText(getToolResult(harness, "bash"))).toContain("Command still running");
		harness.session.dispose();
		await manager.waitForAll();

		expect(aborted).toBe(true);
		expect(
			harness.session.messages.some(
				(message) => message.role === "custom" && message.customType === "bash_completion",
			),
		).toBe(false);
	});

	it.each(["base override", "SDK customTools", "extension registered"] as const)(
		"cancels %s jobs on reload and resubscribes for subsequent completions",
		async (source) => {
			let launches = 0;
			let firstAborted = false;
			let finishSecond!: (result: { exitCode: number | null }) => void;
			const manager = new BashProcessManager();
			const options: BashToolOptions = {
				processManager: manager,
				operations: {
					exec: async (_command, _cwd, { signal }) => {
						launches++;
						return new Promise<{ exitCode: number | null }>((resolve, reject) => {
							if (launches === 1) {
								signal?.addEventListener(
									"abort",
									() => {
										firstAborted = true;
										reject(new Error("aborted"));
									},
									{ once: true },
								);
							} else {
								finishSecond = resolve;
								signal?.addEventListener("abort", () => reject(new Error("aborted")), { once: true });
							}
						});
					},
				},
			};
			const tool = createBashTool(process.cwd(), options);
			const definition = createBashToolDefinition(process.cwd(), options);
			const harness = await createHarness(
				source === "SDK customTools"
					? { customTools: [definition as ToolDefinition] }
					: source === "extension registered"
						? { extensionFactories: [(pi) => pi.registerTool(definition as ToolDefinition)] }
						: { tools: [tool] },
			);
			harnesses.push(harness);
			harness.setResponses([
				fauxAssistantMessage([fauxToolCall("bash", { command: "first", yield_time_ms: 0 })], {
					stopReason: "toolUse",
				}),
				fauxAssistantMessage("first job started"),
			]);

			await harness.session.prompt("start first job");
			await harness.session.reload();
			expect(harness.session.getActiveToolNames()).toContain("bash");
			expect(firstAborted).toBe(true);
			expect(
				harness.session.messages.some(
					(message) => message.role === "custom" && message.customType === "bash_completion",
				),
			).toBe(false);

			const notify = vi.spyOn(harness.session, "sendCustomMessage").mockResolvedValue();
			const secondResult = await (source === "base override" ? tool : wrapToolDefinition(definition)).execute(
				"second-job",
				{ command: "second", yield_time_ms: 0 },
			);
			expect(
				secondResult.content.some((part) => part.type === "text" && part.text.includes("Command still running")),
			).toBe(true);
			finishSecond({ exitCode: 0 });
			await manager.waitForAll();
			expect(launches).toBe(2);
			expect(notify).toHaveBeenCalledOnce();
			expect(notify).toHaveBeenCalledWith(expect.objectContaining({ customType: "bash_completion" }), {
				triggerTurn: true,
				deliverAs: "followUp",
			});
			harness.session.dispose();
			expect(() => manager.onCompletion(() => {})).toThrow("disposed");
		},
	);

	it("owns bash managers registered on session_start even after their definitions are replaced", async () => {
		let registerTool!: (definition: ToolDefinition) => void;
		let finishFirst!: (result: { exitCode: number | null }) => void;
		let replacedAborted = false;
		let secondAborted = false;
		const firstManager = new BashProcessManager();
		const secondManager = new BashProcessManager();
		const first = createBashToolDefinition(process.cwd(), {
			processManager: firstManager,
			operations: {
				exec: async (command, _cwd, { signal }) =>
					new Promise<{ exitCode: number | null }>((resolve, reject) => {
						if (command === "first") {
							finishFirst = resolve;
						} else {
							signal?.addEventListener(
								"abort",
								() => {
									replacedAborted = true;
									reject(new Error("aborted"));
								},
								{ once: true },
							);
						}
					}),
			},
		});
		const second = createBashToolDefinition(process.cwd(), {
			processManager: secondManager,
			operations: {
				exec: async (_command, _cwd, { signal }) =>
					new Promise<{ exitCode: number | null }>((_resolve, reject) => {
						signal?.addEventListener(
							"abort",
							() => {
								secondAborted = true;
								reject(new Error("aborted"));
							},
							{ once: true },
						);
					}),
			},
		});
		const harness = await createHarness({
			extensionFactories: [
				(pi) => {
					registerTool = (definition) => pi.registerTool(definition);
					pi.on("session_start", () => pi.registerTool(first as ToolDefinition));
				},
			],
		});
		harnesses.push(harness);
		await harness.session.bindExtensions({});
		const notify = vi.spyOn(harness.session, "sendCustomMessage").mockResolvedValue();

		expect(
			getMessageText(await wrapToolDefinition(first).execute("first", { command: "first", yield_time_ms: 0 })),
		).toContain("Command still running");
		registerTool(second as ToolDefinition);
		finishFirst({ exitCode: 0 });
		await firstManager.waitForAll();
		expect(notify).toHaveBeenCalledOnce();
		expect(notify).toHaveBeenCalledWith(expect.objectContaining({ customType: "bash_completion" }), {
			triggerTurn: true,
			deliverAs: "followUp",
		});

		expect(
			getMessageText(await wrapToolDefinition(first).execute("replaced", { command: "replaced", yield_time_ms: 0 })),
		).toContain("Command still running");
		expect(
			getMessageText(await wrapToolDefinition(second).execute("second", { command: "second", yield_time_ms: 0 })),
		).toContain("Command still running");
		harness.session.dispose();
		await Promise.all([firstManager.waitForAll(), secondManager.waitForAll()]);
		expect(replacedAborted).toBe(true);
		expect(secondAborted).toBe(true);
		expect(notify).toHaveBeenCalledOnce();
		expect(() => firstManager.onCompletion(() => {})).toThrow("disposed");
		expect(() => secondManager.onCompletion(() => {})).toThrow("disposed");
	});

	it("queues a completion that arrives before its running tool result", async () => {
		let harness!: Harness;
		let finish!: (result: { exitCode: number | null }) => void;
		let completionBeforeResult = false;
		const manager = new BashProcessManager(() => {
			completionBeforeResult = !harness.events.some(
				(event) =>
					event.type === "message_end" && event.message.role === "toolResult" && event.message.toolName === "bash",
			);
		});
		const tool = createBashTool(process.cwd(), {
			processManager: manager,
			operations: {
				exec: async () =>
					new Promise((resolve) => {
						finish = resolve;
					}),
			},
		});
		harness = await createHarness({
			tools: [tool],
			extensionFactories: [
				(pi) => {
					pi.on("tool_execution_end", async (event) => {
						if (event.toolName !== "bash") return;
						finish({ exitCode: 0 });
						await manager.waitForAll();
					});
				},
			],
		});
		harnesses.push(harness);
		harness.setResponses([
			fauxAssistantMessage([fauxToolCall("bash", { command: "race", yield_time_ms: 0 })], {
				stopReason: "toolUse",
			}),
			fauxAssistantMessage("running result received"),
			fauxAssistantMessage("completion received"),
		]);

		await harness.session.prompt("start job");

		const resultIndex = harness.events.findIndex(
			(event) =>
				event.type === "message_end" && event.message.role === "toolResult" && event.message.toolName === "bash",
		);
		const completionIndex = harness.events.findIndex(
			(event) =>
				event.type === "message_end" &&
				event.message.role === "custom" &&
				event.message.customType === "bash_completion",
		);
		expect(completionBeforeResult).toBe(true);
		expect(resultIndex).toBeGreaterThanOrEqual(0);
		expect(completionIndex).toBeGreaterThan(resultIndex);
		expect(getAssistantTexts(harness)).toContain("completion received");
	});

	it.skipIf(process.platform === "win32")("notifies the model when a yielded codemode call finishes", async () => {
		const harness = await createHarness({
			initialActiveToolNames: ["codemode", "bash"],
			extensionFactories: [createCodemodeExtension()],
		});
		harnesses.push(harness);
		const releasePath = join(harness.tempDir, "release-bash");
		const code =
			"const fs = require('node:fs'); setInterval(() => { if (fs.existsSync(process.argv[1])) { console.log('finished'); process.exit(0); } }, 10);";
		const command = `${JSON.stringify(process.execPath)} -e ${JSON.stringify(code)} ${JSON.stringify(releasePath)}`;
		harness.setResponses([
			fauxAssistantMessage(
				[
					fauxToolCall("codemode", {
						code: `return await tools.bash(${JSON.stringify({ command, yield_time_ms: 0 })});`,
					}),
				],
				{ stopReason: "toolUse" },
			),
			fauxAssistantMessage("job started"),
			(context) => {
				const notification = context.messages.find(
					(message) =>
						message.role === "user" &&
						(typeof message.content === "string"
							? message.content.includes("finished")
							: message.content.some((part) => part.type === "text" && part.text.includes("finished"))),
				);
				return fauxAssistantMessage(notification ? "completion received" : "completion missing");
			},
		]);

		await harness.session.prompt("start job");
		const result = getToolResult(harness, "codemode");
		expect(getMessageText(result)).toContain('"status":"running"');
		expect(getAssistantTexts(harness)).toContain("job started");
		expect(getAssistantTexts(harness)).not.toContain("completion received");

		writeFileSync(releasePath, "go");
		await vi.waitFor(() => expect(getAssistantTexts(harness)).toContain("completion received"), {
			timeout: 10_000,
		});
		expect(
			harness.session.messages.some(
				(message) => message.role === "custom" && message.customType === "bash_completion",
			),
		).toBe(true);
	});
});
