/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Chat API base. Defaults to /api, which the dev server proxies. */
  readonly VITE_API_BASE_URL?: string;
  /** Per-user API key, when the project has auth enabled. */
  readonly VITE_API_KEY?: string;
  /** JSON array of projects the admin dashboard watches. See admin/endpoints.ts. */
  readonly VITE_ADMIN_PROJECTS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
