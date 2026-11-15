# nsF5 source-only path correction (2026-11-15)

The first new coefficient-study preparation exits 1 after 1.503 seconds,
before binding sources, constructing real features or fitting any real model.
Its reused JSON helper expects Path objects, while the new source passes a
string. Preserve the original source, attempt, error log and exit receipt.

The separate completion changes only that helper boundary: convert input to
Path before reading. Original features, seeds, roles, models, calibration gate,
numeric guards and original 2,400-second cap remain unchanged. Bind all failed
sources/logs as inputs. Conservatively charge ten seconds for preparation,
failure and diagnosis instead of the original five-second source allowance;
at most 2,390 seconds remain. Never relaunch the failed parent or refit a
consumed real model. Generated controls precede the first actual fit.
