// The stores are exposed on window by StoreBridge for exactly this purpose.
declare global {
  interface Window {
    __stores: {
      model: { getState(): { status: string } };
      trace: { getState(): { status: string; loss: number | null } };
      learn: { getState(): { markPassed(slug: string): void; passed: string[] } };
    };
  }
}
export {};
