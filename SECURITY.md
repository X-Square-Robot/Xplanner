# Security

Do not commit API keys, access tokens, private endpoints, credentials, or private dataset URLs.
Use environment variables or an untracked `.env` file based on `.env.example`.

If a credential is committed, revoke or rotate it first. Removing it in a later commit is not
sufficient because the value remains available in Git history. Rewrite the affected public
history and verify all server-side merge-request and cached refs before publishing the project.

Report security issues privately to the repository maintainers rather than opening a public issue.
