import { DependencyList, useEffect, useState } from "react";

interface AsyncState<T> {
  data: T | null;
  error: string;
  loading: boolean;
}

export function useAsyncResource<T>(
  loader: () => Promise<T>,
  deps: DependencyList,
): AsyncState<T> & { reload: () => Promise<void> } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    setError("");
    try {
      const next = await loader();
      setData(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown request error");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, error, loading, reload: load };
}

export function usePollingResource<T>(
  loader: () => Promise<T>,
  deps: DependencyList,
  intervalMs = 2500,
): AsyncState<T> & { reload: () => Promise<void> } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    setError("");
    try {
      const next = await loader();
      setData(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown request error");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    const id = window.setInterval(() => {
      void load();
    }, intervalMs);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, ...deps]);

  return { data, error, loading, reload: load };
}
