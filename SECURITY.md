# Security policy

SegAgent is a research application implemented in `segagent/`. It
does not provide an authentication or authorization layer and is intended for
local, controlled research use. Bind services to loopback by default. Public hosting
requires a separate security assessment and appropriate access controls; CORS
configuration is not authentication.

Do not upload protected health information (PHI), patient images, credentials,
or identifying screenshots to GitHub issues, pull requests, or public logs.
Use synthetic data when demonstrating a problem. Keep local case stores,
SQLite checkpoints, configuration secrets, model weights, and trace exports out
of version control. Review tracing and telemetry settings before using any
sensitive input.

## Report a vulnerability

If GitHub private vulnerability reporting is enabled for this repository, use
[Report a vulnerability](https://github.com/Syndr0/SegAgent/security/advisories/new)
to contact the maintainers privately. If that option is unavailable, contact a
repository maintainer through a private contact method they publish on their
GitHub profile. Do not include exploit details or sensitive data in a public
issue while arranging private contact.

Provide the affected commit, component, expected impact, and minimal
reproduction steps using synthetic data. If you are reporting a finding in the
historical `archive/prototype/` source, identify that path explicitly; it is not an
alternative supported application. Redact secrets, local identifiers, and
patient information from attachments and logs.

The repository does not currently publish a supported-version matrix or a
guaranteed security response time. Include your exact commit so maintainers can
assess the affected code accurately.
