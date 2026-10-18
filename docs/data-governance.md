# Local data access and retention

The application uses the operating-system account as its access boundary.
Run it under a dedicated, non-shared account. These controls do not provide
application login, separate roles for people sharing an account, encryption,
or protection from administrators who can take ownership.

## Access controls

New output, event, log and enrollment directories receive private permissions
before the application writes data into them. Existing directories are not
silently changed when exporting into a folder; migrate managed data explicitly:

```powershell
integrated-monitoring --secure-data
```

For a source checkout use `python src/main.py --secure-data`.
Run this as the account that will operate the application, with monitoring
stopped. It restricts the configured output, enrollment, log and event roots
and their existing contents. Never run it under a temporary sandbox account
or a different administrator account: that account would become the owner.

On Windows, directories receive protected ACLs granting FullControl only to
the current account and SYSTEM, with inheritance for new files. Existing
files receive the same restricted grants. On POSIX, directories use 0700 and
files use 0600 during migration. New files are protected by their private
parent directory. ACL/chmod failures stop the operation; symlinks, Windows
reparse points and multiply-linked files are rejected. Migration may have
partially tightened permissions if an OS operation fails; correct the failure
and rerun it under the same account.

On macOS, descendants of the root-owned `/var -> /private/var` and
`/tmp -> /private/tmp` system aliases are supported. The alias target and
ownership are checked, then the complete physical path is checked again.
The aliases themselves cannot be managed roots; user-created links and
links beneath the physical path remain rejected.

Custom CLI export folders outside the configured output root are not included
in migration or retention. Use a dedicated private folder or `MONITOR_HOME`
and keep exports under its output folder. Original input media, validation
reports, backups, cloud copies and external enrollment copies require their
own access/retention controls. Do not point output at a shared or general-use
directory.

## Retention policy

Defaults are operational starting points, not a legal retention determination.
Set positive integer environment values before launch:

| Data | Environment variable | Default | Eligible files |
| --- | --- | --- | --- |
| Annotated recordings/images | `MONITOR_RECORDING_RETENTION_DAYS` | 30 days | Supported media whose stem ends in `_annotated`, beneath configured output |
| Enrollment identity images | `MONITOR_IDENTITY_RETENTION_DAYS` | 365 days | JPG/JPEG/PNG/BMP/WEBP beneath enrollments |
| Event and application logs | `MONITOR_LOG_RETENTION_DAYS` | 30 days | JSONL under events; monitoring.log and numbered rotations under logs |

Age is measured from filesystem modification time, not last recognition,
capture time or consent expiry. Copying/restoring files can change this clock.
Original recordings are deliberately excluded from automated deletion.
Partial export files are excluded and require manual review. Deleting an
identity's enrollment images does not erase that person's appearances or name
from older recordings/logs; those expire under their respective policies.

Preview without deletion:

```powershell
integrated-monitoring --retention
```

Review the displayed paths and stop all writers, then explicitly apply:

```powershell
integrated-monitoring --retention --apply-retention
```

The apply command calculates a fresh plan, validates files and containment,
then unlinks only eligible files. It rejects changes detected before deletion
and never recursively deletes trees. Empty enrollment parent directories are
removed after their expired images, so identity folder names are not retained.
Filesystem unlink is not secure erasure and does not delete backup copies.
This is maintenance for private, quiescent directories, not a race-proof
security boundary against another process running as the same account.

No background deletion or scheduled task is enabled. To enforce the desired
time limit operationally, run maintenance regularly under the application
account while monitoring is stopped. The console preview itself contains
sensitive paths; protect any saved copies of its output.

## Verification on this workstation

The managed output/enrollment/event/log trees were restricted under Jett's
Windows account. The initial retention preview selected zero files; no user
data was deleted. Tests cover expiry selection, changed-file refusal, links,
permission failures and Windows ACL principals. POSIX permission behavior and
deployment backup/retention procedures still require target-platform checks.
