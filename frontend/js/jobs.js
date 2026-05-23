/**
 * Job manager — subscribe to async pipeline jobs with progress tracking.
 * Provides a toast-style progress bar for detect/ocr/correct/inpaint/render.
 */
const JobManager = {
    _activeJobs: {},

    /**
     * Submit a job and auto-subscribe to progress events.
     * @param {string} url - POST endpoint (e.g. "/api/projects/{id}/jobs/correct")
     * @param {object} options - { onProgress, onComplete, onError }
     * @returns {Promise}
     */
    async submit(url, options = {}) {
        const { onProgress, onComplete, onError } = options;

        try {
            const resp = await fetch(url, { method: "POST" });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ detail: resp.statusText }));
                if (onError) onError(err);
                return null;
            }
            const job = await resp.json();
            const jobId = job.job_id;

            this._activeJobs[jobId] = { ...job, onProgress, onComplete, onError };
            this._listenToEvents(jobId);
            return job;
        } catch (e) {
            if (onError) onError({ detail: e.message });
            return null;
        }
    },

    async _listenToEvents(jobId) {
        const info = this._activeJobs[jobId];
        if (!info) return;

        try {
            const resp = await fetch(`/api/jobs/${jobId}/events`, {
                headers: { "Accept": "text/event-stream" },
            });
            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                const lines = buffer.split("\n");
                buffer = lines.pop() || "";

                for (const line of lines) {
                    if (!line.startsWith("data: ")) continue;
                    try {
                        const event = JSON.parse(line.slice(6));
                        if (info.onProgress && event.type === "progress") {
                            info.onProgress(event.completed, event.total);
                        }
                        if (event.type === "complete" || event.type === "done") {
                            if (info.onComplete) info.onComplete(event);
                            delete this._activeJobs[jobId];
                            return;
                        }
                        if (event.type === "error" || event.type === "cancelled") {
                            if (info.onError) info.onError(event);
                            delete this._activeJobs[jobId];
                            return;
                        }
                    } catch (e) {
                        // skip non-JSON lines
                    }
                }
            }
        } catch (e) {
            if (info.onError) info.onError({ detail: "SSE connection lost" });
        }
        delete this._activeJobs[jobId];
    },

    /** Cancel a running job. */
    async cancel(jobId) {
        try {
            await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
        } catch (e) {
            console.warn("Cancel failed:", e);
        }
    },

    /** Get active job count. */
    count() {
        return Object.keys(this._activeJobs).length;
    },
};
