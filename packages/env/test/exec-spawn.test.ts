import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import { getOrThrow } from "@earendil-works/pi-durable/env";
import { afterAll, describe, expect, it, vi } from "vitest";
import { Connection, type RequestOptions } from "../src/connection.ts";
import { RemoteExecutionEnv } from "../src/remote-env.ts";
import { daemon } from "./daemon.ts";

const dirs: string[] = [];
const connections: Connection[] = [];
function environment() {
	const cwd = mkdtempSync(join(tmpdir(), "pi-env-spawn-"));
	dirs.push(cwd);
	const connection = new Connection({ command: [daemon] });
	connections.push(connection);
	return { cwd, env: new RemoteExecutionEnv({ connection, id: "remote:test", cwd }) };
}
afterAll(() => {
	for (const connection of connections) connection.close();
	for (const dir of dirs) rmSync(dir, { recursive: true, force: true });
});

describe("remote spawned command", () => {
	it("refuses a daemon advertising the older wire protocol", async () => {
		const script = `
			process.stdout.write("PI-ENV " + process.argv.at(-1) + "\\n");
			process.stdin.once("data", () => {
				const json = Buffer.from(JSON.stringify({ protocol: 1 }));
				const header = Buffer.alloc(13);
				header.writeUInt32BE(9 + json.length, 0);
				header.writeUInt8(2, 4);
				header.writeUInt32BE(1, 5);
				header.writeUInt32BE(json.length, 9);
				process.stdout.write(Buffer.concat([header, json]));
			});
		`;
		const connection = new Connection({ command: [process.execPath, "-e", script] });
		connections.push(connection);
		await expect(connection.info()).rejects.toMatchObject({
			code: "unknown",
			message: "Unsupported protocol 1",
		});
	});

	it.skipIf(process.platform === "win32")("forwards the daemon's child PID, not its transport PID", async () => {
		const { env } = environment();
		const transport = (await env.connection.info()).pid;
		const calls: number[] = [];
		let output = "";
		const result = getOrThrow(
			await env.exec(
				'printf "%s" "$$"',
				{
					onSpawn: (pid) => {
						if (pid === undefined) throw new Error("Remote did not report its spawned PID");
						calls.push(pid);
					},
					onOutput: (text) => {
						output += text;
					},
				},
				BACKGROUND_CONTEXT,
			),
		);
		expect(result.exitCode).toBe(0);
		expect(calls).toHaveLength(1);
		expect(Number.isSafeInteger(calls[0]) && calls[0]! > 0).toBe(true);
		expect(calls[0]).not.toBe(transport);
		expect(Number(output)).toBe(calls[0]);
	});

	it("awaits an asynchronous spawn observer even if the remote command exits first", async () => {
		const { env } = environment();
		let release!: () => void;
		const gate = new Promise<void>((resolve) => {
			release = resolve;
		});
		let started!: () => void;
		const observed = new Promise<void>((resolve) => {
			started = resolve;
		});
		let settled = false;
		const execution = env
			.exec(
				["sh", "-c", "exit 0"],
				{
					onSpawn: async () => {
						started();
						await gate;
					},
				},
				BACKGROUND_CONTEXT,
			)
			.then((result) => {
				settled = true;
				return result;
			});
		await observed;
		await new Promise((resolve) => setTimeout(resolve, 150));
		expect(settled).toBe(false);
		release();
		expect(getOrThrow(await execution).exitCode).toBe(0);
	});

	it.skipIf(process.platform === "win32")(
		"holds command output behind a slow startup callback without blocking other requests",
		async () => {
			const { env, cwd } = environment();
			const size = 1_500_000;
			const path = join(cwd, "complete.log");
			writeFileSync(path, "");
			let release!: () => void;
			const gate = new Promise<void>((resolve) => {
				release = resolve;
			});
			let started!: () => void;
			const observed = new Promise<void>((resolve) => {
				started = resolve;
			});
			let output = "";
			const execution = env.exec(
				[process.execPath, "-e", `process.stdout.write('x'.repeat(${size}))`],
				{
					spill: { path, afterBytes: size * 2, afterLines: size * 2 },
					onSpawn: async () => {
						started();
						await gate;
					},
					onOutput: (text) => {
						output += text;
					},
				},
				BACKGROUND_CONTEXT,
			);
			await observed;
			try {
				expect(
					getOrThrow(await env.exec(["sh", "-c", "printf other"], undefined, BACKGROUND_CONTEXT)).exitCode,
				).toBe(0);
				await new Promise((resolve) => setTimeout(resolve, 200));
				expect(output).toBe("");
				expect(readFileSync(path, "utf8")).toBe("");
			} finally {
				release();
			}
			expect(getOrThrow(await execution).exitCode).toBe(0);
			expect(output).toBe("x".repeat(size));
			expect(readFileSync(path, "utf8")).toBe(output);
		},
	);

	it.skipIf(process.platform === "win32")(
		"keeps the hard timeout armed through a fast exit and delayed startup callback",
		async () => {
			const { env } = environment();
			let started = false;
			const result = await env.exec(
				["sh", "-c", "exit 0"],
				{
					timeout: 0.1,
					onSpawn: async () => {
						started = true;
						await new Promise((resolve) => setTimeout(resolve, 250));
					},
				},
				BACKGROUND_CONTEXT,
			);
			expect(started).toBe(true);
			expect(result).toMatchObject({ ok: false, error: { code: "timeout" } });
		},
	);

	it.skipIf(process.platform === "win32")(
		"kills descendants holding pipes after a fast shell exit and delayed timeout",
		async () => {
			const { env, cwd } = environment();
			const result = await env.exec(
				["sh", "-c", "(sleep 0.5; touch marker) &"],
				{
					timeout: 0.1,
					onSpawn: async () => {
						await new Promise((resolve) => setTimeout(resolve, 250));
					},
				},
				BACKGROUND_CONTEXT,
			);
			expect(result).toMatchObject({ ok: false, error: { code: "timeout" } });
			await new Promise((resolve) => setTimeout(resolve, 600));
			expect(getOrThrow(await env.exists(join(cwd, "marker"), BACKGROUND_CONTEXT))).toBe(false);
		},
	);

	it("does not call the observer on startup failure", async () => {
		const { env, cwd } = environment();
		const onSpawn = vi.fn();
		expect(await env.exec([join(cwd, "missing")], { onSpawn }, BACKGROUND_CONTEXT)).toMatchObject({
			ok: false,
			error: { code: "spawn_error" },
		});
		expect(onSpawn).not.toHaveBeenCalled();
	});

	it.each(["sync", "async"])(
		"kills and reaps on %s observer failure without cancelling other commands",
		async (mode) => {
			const { env, cwd } = environment();
			const path = join(cwd, "log");
			writeFileSync(path, "");
			const other = env.exec(["sh", "-c", "sleep 0.3; exit 0"], undefined, BACKGROUND_CONTEXT);
			const result = await env.exec(
				"sleep 1; touch marker",
				{
					spill: { path, afterBytes: 1000, afterLines: 1000 },
					onSpawn:
						mode === "sync"
							? () => {
									throw new Error("observer failed");
								}
							: async () => {
									await Promise.resolve();
									throw new Error("observer failed");
								},
				},
				BACKGROUND_CONTEXT,
			);
			expect(result).toMatchObject({
				ok: false,
				error: { code: "callback_error", message: "observer failed", spillPath: path },
			});
			expect(getOrThrow(await other).exitCode).toBe(0);
			expect(getOrThrow(await env.exists(join(cwd, "marker"), BACKGROUND_CONTEXT))).toBe(false);
		},
	);

	it("appends an empty and a large complete log, and preserves a timed-out log", async () => {
		const { env, cwd } = environment();
		const path = join(cwd, "log");
		writeFileSync(path, "prefix:");
		const spill = { path, afterBytes: 100_000_000, afterLines: 100_000_000 };
		expect(getOrThrow(await env.exec(["sh", "-c", "exit 0"], { spill }, BACKGROUND_CONTEXT)).spillPath).toBe(path);
		expect(readFileSync(path, "utf8")).toBe("prefix:");
		const size = 1_500_000;
		expect(
			getOrThrow(
				await env.exec(
					[process.execPath, "-e", `process.stdout.write('a'.repeat(${size})); process.stderr.write('b')`],
					{ spill },
					BACKGROUND_CONTEXT,
				),
			).spillPath,
		).toBe(path);
		expect(readFileSync(path, "utf8").length).toBe(size + 8);
		expect(readFileSync(path, "utf8")).toContain("b");
		const timeout = await env.exec(["sh", "-c", "printf tail; sleep 5"], { spill, timeout: 0.3 }, BACKGROUND_CONTEXT);
		expect(timeout).toMatchObject({ ok: false, error: { code: "timeout", spillPath: path } });
		expect(readFileSync(path, "utf8")).toContain("tail");
	}, 30_000);

	it("fails to open a missing log before spawning", async () => {
		const { env, cwd } = environment();
		const result = await env.exec(
			["sh", "-c", "touch marker"],
			{
				spill: { path: join(cwd, "missing"), afterBytes: 5, afterLines: 5 },
				onSpawn: () => {
					throw new Error("should not spawn");
				},
			},
			BACKGROUND_CONTEXT,
		);
		expect(result).toMatchObject({ ok: false, error: { code: "unknown" } });
		expect(getOrThrow(await env.exists("marker", BACKGROUND_CONTEXT))).toBe(false);
	});

	it.each(["invalid", "duplicate"])("cancels and joins a real command after a %s spawned event", async (kind) => {
		const { env } = environment();
		const connection = env.connection;
		await connection.info();
		const originalRequest = connection.request.bind(connection);
		const spawned = Promise.withResolvers<void>();
		const observer = Promise.withResolvers<void>();
		let backendSettled = false;
		vi.spyOn(connection, "request").mockImplementation((op, json, options: RequestOptions = {}) => {
			const request = originalRequest(
				op,
				json,
				op === "exec"
					? {
							...options,
							onEvent: (event, payload) => {
								if (event.kind === "spawned") {
									spawned.resolve();
									options.onEvent?.(kind === "invalid" ? { ...event, pid: -1 } : event, payload);
									if (kind === "duplicate") options.onEvent?.(event, payload);
								} else {
									options.onEvent?.(event, payload);
								}
							},
						}
					: options,
			);
			if (op === "exec") {
				void request.then(
					() => {
						backendSettled = true;
					},
					() => {
						backendSettled = true;
					},
				);
			}
			return request;
		});
		const onSpawn = vi.fn(() => observer.promise);
		const execution = env.exec(["sh", "-c", "exit 0"], { onSpawn }, BACKGROUND_CONTEXT);
		try {
			await spawned.promise;
			await new Promise((resolve) => setTimeout(resolve, 350));
			expect(backendSettled).toBe(true);
		} finally {
			observer.resolve();
			if (!backendSettled) connection.close();
		}
		const result = await execution;
		expect(result).toMatchObject({ ok: false, error: { code: "unknown" } });
		expect(onSpawn).toHaveBeenCalledTimes(kind === "invalid" ? 0 : 1);
	});
});
