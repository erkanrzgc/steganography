# Cloud AI removal

Cloud AI triage was an explanation layer, not a validated detector. It has been
removed from the application and package extras. Existing native analysis,
embedding, extraction, CTF, evidence, and report workflows remain available.
Local model/research code and historical results are retained, not promoted.

## CLI and configuration

Remove `--ai` and `--allow-ai-file-upload` from scripts. They are hidden from
help and rejected with a clear error instead of silently performing a local
scan under an AI-labelled command. The `ai` installation extra is removed.
NIM and other cloud-provider settings are no longer used; remove obsolete
credentials from your local `.env` and revoke unused provider keys.

## API and Python compatibility

Existing `/v1` and `/v2` routes and report schemas remain. Default local requests
retain their behavior. Legacy v1 `ai=true` or `allow_ai_file_upload=true`
requests return **410 Gone**, before application analysis, upload-directory
creation or persisted scan jobs. Framework multipart parsing may still spool
the request body temporarily; this is not a pre-body network rejection.
False/default values remain accepted; retired parameters are hidden from OpenAPI.

`APISettings.allow_ai_file_upload` remains an inert constructor field for old
callers and cannot enable uploads to a provider. Its environment setting is no
longer read. `AnalysisService(ai_provider=None)` remains accepted; passing a
provider raises a clear error and never calls it. A retired `ai_triage` plug-in
is not invoked. Other `Analyzer.analyze(Path)` plug-ins remain compatible.

Historical `ai_triage` results remain excluded from primary scoring and findings.
Published benchmark artifacts are not rewritten to erase the old experiment
history. This removal does not claim an increase in detection accuracy.
