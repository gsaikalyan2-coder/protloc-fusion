/// <reference types="vite/client" />

/**
 * Build-time configuration injected by Vite.
 *
 * Declared explicitly rather than relying on the ambient default so that a
 * missing variable is a type error at the point of use instead of an `any` that
 * silently spreads through the API client.
 */
interface ImportMetaEnv {
  /** Origin of the backend in production; unset in development, where Vite proxies /api. */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
