import { mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import { afterEach, describe, expect, it, vi } from "vitest";
import { getOrThrow } from "../src/env/index.ts";
import { NodeExecutionEnv } from "../src/env/node.ts";

const dirs: string[] = [];
function environment() {
	const cwd = mkdtempSync(join(tmpdir(), "pi-env-spawn-"));
	dirs.push(cwd);
	return { env: new NodeExecutionEnv({ cwd }), cwd };
}
afterEach(() => {
	for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe("NodeExecutionEnv spawn observer and complete log", () => {
	it.skipIf(process.platform === "win32")("creates private log files even under a permissive umask", async () => {
		const { env } = environment();
		const path = getOrThrow(await env.createTempFile({ suffix: ".log" }, BACKGROUND_CONTEXT));
		dirs.push(dirname(path));
		expect(statSync(path).mode & 0o777).toBe(0o600);
	});
	it.skipIf(process.platform === "win32")("reports the actual shell process PID once, before output", async () => {
		const { env } = environment();
		const calls: number[] = [];
		let release!: () => void;
		const gate = new Promise<void>((resolve) => {
			release = resolve;
		});
		let started!: () => void;
		const observed = new Promise<void>((resolve) => {
			started = resolve;
		});
		let text = "";
		const execution = env.exec(
			'printf "%s" "$$"',
			{
				onSpawn: async (pid) => {
					if (pid === undefined) throw new Error("Node did not report its spawned PID");
					calls.push(pid);
					started();
					await gate;
				},
				onOutput: (chunk) => {
					text += chunk;
				},
			},
			BACKGROUND_CONTEXT,
		);
		await observed;
		expect(calls).toHaveLength(1);
		expect(Number.isSafeInteger(calls[0]) && calls[0]! > 0).toBe(true);
		expect(text).toBe("");
		// Let the shell exit while its pipe is paused; post-exit stdio grace must not discard its output.
		await new Promise((resolve) => setTimeout(resolve, 200));
		expect(text).toBe("");
		release();
		expect(getOrThrow(await execution).exitCode).toBe(0);
		expect(Number(text)).toBe(calls[0]);
	});

	it.skipIf(process.platform === "win32")(
		"keeps the hard timeout armed through a fast exit and delayed startup callback",
		async () => {
			const { env } = environment();
			let started = false;
			const result = await env.exec(
				"exit 0",
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

	it.skipIf(process.platform === "win32")("does not signal a reaped child PID after a delayed timeout", async () => {
		const { env } = environment();
		let pid: number | undefined;
		const originalKill = process.kill.bind(process);
		const kill = vi.spyOn(process, "kill").mockImplementation((target, signal) => {
			if (pid !== undefined && target === -pid)
				throw Object.assign(new Error("No process group"), { code: "ESRCH" });
			if (pid !== undefined && target === pid) return true;
			return originalKill(target, signal);
		});
		try {
			const result = await env.exec(
				"exit 0",
				{
					timeout: 0.1,
					onSpawn: async (spawnedPid) => {
						pid = spawnedPid;
						await new Promise((resolve) => setTimeout(resolve, 250));
					},
				},
				BACKGROUND_CONTEXT,
			);
			expect(result).toMatchObject({ ok: false, error: { code: "timeout" } });
			expect(pid).toBeDefined();
			expect(kill).not.toHaveBeenCalledWith(pid, "SIGKILL");
		} finally {
			kill.mockRestore();
		}
	});

	it.skipIf(process.platform === "win32")(
		"kills descendants that still hold pipes after the shell exits",
		async () => {
			const { env, cwd } = environment();
			const result = await env.exec(
				"(sleep 0.5; touch marker) &",
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

	it("does not observe startup failures", async () => {
		const { env, cwd } = environment();
		let calls = 0;
		const onSpawn = () => {
			calls++;
		};
		expect(await env.exec([join(cwd, "missing")], { onSpawn }, BACKGROUND_CONTEXT)).toMatchObject({
			ok: false,
			error: { code: "spawn_error" },
		});
		expect(await env.exec([], { onSpawn }, BACKGROUND_CONTEXT)).toMatchObject({
			ok: false,
			error: { code: "spawn_error" },
		});
		expect(calls).toBe(0);
	});

	it.each(["sync", "async"])("kills only its command and waits after a %s observer failure", async (mode) => {
		const { env, cwd } = environment();
		const path = join(cwd, "log");
		writeFileSync(path, "");
		const other = env.exec(
			[process.execPath, "-e", "setTimeout(() => process.stdout.write('alive'), 300)"],
			undefined,
			BACKGROUND_CONTEXT,
		);
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
		expect(await env.exists(join(cwd, "marker"), BACKGROUND_CONTEXT)).toEqual({ ok: true, value: false });
	});

	it("opens an existing log before spawn and records empty and large combined output without thresholds", async () => {
		const { env, cwd } = environment();
		const path = join(cwd, "log");
		writeFileSync(path, "prefix:");
		const empty = getOrThrow(
			await env.exec(
				[process.execPath, "-e", ""],
				{ spill: { path, afterBytes: 1_000_000, afterLines: 1_000_000 } },
				BACKGROUND_CONTEXT,
			),
		);
		expect(empty.spillPath).toBe(path);
		expect(readFileSync(path, "utf8")).toBe("prefix:");
		const size = 2_000_000;
		const result = getOrThrow(
			await env.exec(
				[process.execPath, "-e", `process.stdout.write('a'.repeat(${size})); process.stderr.write('b')`],
				{ spill: { path, afterBytes: 1_000_000_000, afterLines: 1_000_000_000 } },
				BACKGROUND_CONTEXT,
			),
		);
		expect(result.spillPath).toBe(path);
		const content = readFileSync(path, "utf8");
		expect(content.length).toBe(size + 8);
		expect(content).toContain("b");
		expect(content.split("a").length - 1).toBe(size);
	});

	it("fails logging before spawn and preserves the log path on timeout", async () => {
		const { env, cwd } = environment();
		const missing = join(cwd, "not-created");
		let observed = false;
		expect(
			await env.exec(
				"touch marker",
				{
					spill: { path: missing, afterBytes: 10, afterLines: 10 },
					onSpawn: () => {
						observed = true;
					},
				},
				BACKGROUND_CONTEXT,
			),
		).toMatchObject({ ok: false, error: { code: "unknown" } });
		expect(observed).toBe(false);
		expect(getOrThrow(await env.exists("marker", BACKGROUND_CONTEXT))).toBe(false);
		mkdirSync(join(cwd, "nested"));
		const path = join(cwd, "nested", "log");
		writeFileSync(path, "");
		const result = await env.exec(
			"printf 'before'; sleep 5",
			{ timeout: 0.2, spill: { path, afterBytes: 999, afterLines: 999 } },
			BACKGROUND_CONTEXT,
		);
		expect(result).toMatchObject({ ok: false, error: { code: "timeout", spillPath: path } });
		expect(readFileSync(path, "utf8")).toBe("before");
	});
});
