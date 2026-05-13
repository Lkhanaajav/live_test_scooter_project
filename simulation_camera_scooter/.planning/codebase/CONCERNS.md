# Codebase Concerns

**Analysis Date:** 2026-03-04

## Tech Debt

**Bare exception handling and silent failures:**
- Issue: Multiple exception handlers use `except Exception:` or `pass` without logging or recovery strategy
- Files: `live_heading_demo.py` (lines 387, 399, 403, 509), `realtime_nav_core.py` (lines 358-362)
- Impact: Silent failures in GPS reading, serial communication, and skeltonization make debugging extremely difficult; user gets no feedback when critical systems fail
- Fix approach: Replace bare exception handlers with specific exception types; log errors with descriptive messages; propagate exceptions up to main loop with user feedback

**Missing error recovery in real-time pipeline:**
- Issue: When SegFormer model loading fails (line 190-192 in `fast_road_detector.py`), exception is raised but caller may not handle gracefully
- Files: `fast_road_detector.py` (lines 190-192), `live_heading_demo.py` (lines 1230-1238)
- Impact: Pipeline crashes entirely if model fails to load; no fallback to CPU or cached model
- Fix approach: Implement graceful degradation - try GPU, fall back to CPU; cache model path; allow manual retry

**Hardcoded camera calibration matrix:**
- Issue: BEV perspective transform matrix hardcoded as fixed numpy array in multiple files
- Files: `camera_waypoint_pipeline.py` (lines 23-39), `live_heading_demo.py` (calibration loaded from `bev_calibration.npy`)
- Impact: Works only for one specific camera setup; requires manual recalibration for different camera angles; no validation that loaded calibration is correct
- Fix approach: Store calibration in standard format (JSON); validate matrix properties before use; add warning if using default vs. loaded calibration

**Print-based logging instead of proper logging module:**
- Issue: Codebase uses `print()` for all diagnostics; no log levels, no timestamps in debug output, no way to filter or route logs
- Files: All Python files use `print()` throughout, `live_heading_demo.py` (182+ print statements)
- Impact: Impossible to capture logs programmatically; stdout pollution during normal operation; difficult to correlate events across modules
- Fix approach: Replace all print statements with logging module; use proper log levels (DEBUG, INFO, WARNING, ERROR); configure handlers

## Known Bugs

**Path discontinuity detection may fail with small lookahead:**
- Symptoms: When path changes suddenly (T-junction branch switch), controller may accept discontinuous path if discontinuity threshold is exceeded
- Files: `realtime_nav_core.py` (lines 836-848)
- Trigger: Sharp turns, narrow junctions where path endpoints differ significantly in both lateral and heading
- Workaround: Increase `path_discont_lat_m` and `path_discont_head_deg` thresholds; test with known problematic junctions
- Note: Current defaults (0.45m lateral, 25° heading) may be too permissive for sharp turns

**Temporal smoothing breaks on frame drops:**
- Symptoms: When frame skipping occurs (stride > 1 or dropped frames), temporal filters apply wrong weights because frame timestamps are not tracked
- Files: `fast_road_detector.py` (lines 194-206, 317-318), `live_heading_demo.py` (mask smoothing)
- Trigger: Any configuration with `frame_step > 1` or intermittent frame capture
- Workaround: Only use stride=1 for accurate temporal filtering; disable smoothing when using frame skipping
- Fix approach: Track timestamps for each frame; compute correct filter weights based on actual time deltas, not frame count

**GPS waypoint matching uses distance only:**
- Symptoms: Robot may overshoot waypoints or get stuck near waypoint if GPS accuracy is poor (±5m error bands)
- Files: `live_heading_demo.py` (lines 436-451)
- Trigger: Outdoors with poor GPS fix quality; happens regularly with consumer GPS modules
- Workaround: Reduce waypoint proximity threshold below minimum GPS error
- Fix approach: Weight waypoint distance by GPS fix quality (HDOP); require N consecutive good fixes before advancing

**Mask upsampling mismatch in camera pipeline:**
- Symptoms: When inference is done at different resolution than camera, mask resizing can cause misalignment
- Files: `camera_waypoint_pipeline.py` (lines 272-283), `live_heading_demo.py` (complex resizing logic throughout)
- Trigger: Using `resize_w` / `resize_h` arguments; different model input sizes
- Workaround: Always use native camera resolution for inference
- Fix approach: Store inference resolution with mask; validate shape before BEV transform; add debug visualization

## Security Considerations

**Serial port data not validated:**
- Risk: Robot accepts steering commands from serial port without checksum or authentication
- Files: `live_heading_demo.py` (lines 544-548), `camera_waypoint_pipeline.py` (no serial validation)
- Current mitigation: Only listening on explicitly opened port; assumes physical isolation
- Recommendations: Add frame checksums to serial protocol; validate command ranges; log all sent/received commands for audit

**Model files loaded without verification:**
- Risk: Model directory paths are constructed from user input; could potentially load wrong model
- Files: `fast_road_detector.py` (lines 154-192), `live_heading_demo.py` (lines 42-46)
- Current mitigation: None - model is loaded from config path without validation
- Recommendations: Verify model directory contains expected config files; check model hash against known good values; fail explicitly if model corrupted

**GPS data not validated before use:**
- Risk: Malformed NMEA sentences could cause parsing errors or invalid coordinates
- Files: `camera_waypoint_pipeline.py` (lines 107-119), `live_heading_demo.py` (lines 340-372)
- Current mitigation: Try/except catches some errors, but returns None without logging
- Recommendations: Validate latitude/longitude ranges (-90 to 90, -180 to 180); validate fix quality before using for navigation; log parse failures

## Performance Bottlenecks

**SegFormer inference dominates pipeline:**
- Problem: Semantic segmentation takes ~40-60ms per frame on typical GPU
- Files: `fast_road_detector.py` (line 303), `live_heading_demo.py` (1400-1600 series)
- Cause: Full-resolution inference; no frame-skipping optimization built in
- Improvement path: Implement adaptive inference (detect easy frames, skip hard ones); use model quantization; lower input resolution when FPS drops

**Graph search in BEVPathExtractor has unbounded complexity:**
- Problem: DFS with branch limit (3) but no depth-first pruning; can explore O(3^depth) paths
- Files: `realtime_nav_core.py` (lines 566-659)
- Cause: While loop processes all generated branches; no early termination when good candidate found
- Improvement path: Sort candidates by cost before expansion; stop search when cost improvement drops below threshold; implement A* instead of DFS

**Temporal smoothing done every frame regardless of path quality:**
- Problem: Exponential filter runs even when path is invalid or discontinuous
- Files: `realtime_nav_core.py` (lines 890-892), `fast_road_detector.py` (lines 316-318)
- Cause: Filter state persists across bad paths; no reset on discontinuity
- Improvement path: Reset filter state when path changes; weight smoothing by path confidence; skip smoothing for invalid paths

**Memory accumulation from logging in long runs:**
- Problem: CSV logger appends every frame; 1-hour run = 28,800 rows; memory not freed until close
- Files: `live_heading_demo.py` (lines 174-210)
- Cause: No row limit or buffer flush strategy
- Improvement path: Implement circular buffer for last N rows; flush to disk periodically; compress old CSV files

## Fragile Areas

**BEV calibration hot-path:**
- Files: `live_heading_demo.py` (lines 752-810, 820-827)
- Why fragile: Single 4-point perspective transform controls entire geometric pipeline; if calibration is slightly off, all downstream geometry is wrong; no validation that 4 points form valid perspective
- Safe modification: Always validate that source/destination points are not collinear; check that perspective matrix is invertible; test calibration on known-distance objects
- Test coverage: No automated tests for calibration accuracy; only manual visual inspection

**Medial axis skeleton extraction:**
- Files: `realtime_nav_core.py` (lines 357-387, 404-454)
- Why fragile: Thinning algorithm (Guo-Hall or fallback) is sensitive to noise and preprocessing; disconnections in skeleton cause graph nodes to disappear
- Safe modification: Increase preprocessing morphology kernel sizes; test on various sidewalk textures; add post-processing to reconnect broken skeletons
- Test coverage: No unit tests for skeleton quality; no visualization of intermediate skeleton steps in production

**Pure pursuit controller feedback loop:**
- Files: `realtime_nav_core.py` (lines 850-907)
- Why fragile: Relies on path model state (`prev_delta_rad`, `prev_kappa_ref`); discontinuous path updates can cause large steering jerks
- Safe modification: Add path transition logic before updating model; gradually interpolate between old and new steering commands; validate lookahead point is forward of ego
- Test coverage: No tests for edge cases (path loop-back, sharp U-turns, missing paths)

**Frame stabilization in live_heading_demo:**
- Files: `live_heading_demo.py` (lines 1262-1264, 1414-1449)
- Why fragile: Implements geometric center-of-motion tracking without velocity prediction; can create jitter or drift on static scenes
- Safe modification: Add velocity history; validate stabilized point is within frame; disable on low-motion frames
- Test coverage: No motion tracking validation; no automated tests on video with known motion

## Scaling Limits

**Single-threaded frame processing:**
- Current capacity: ~8-10 FPS real-time on Rock 5B (mentioned in realtime_nav_core.py docstring)
- Limit: SegFormer + YOLOv8 inference are sequential; GPU underutilized during CPU processing
- Scaling path: Implement producer-consumer pattern with separate inference threads; queue-based architecture; process multiple frames in parallel

**In-memory model storage:**
- Current capacity: ~1 perspective matrix (96 bytes) + 1-2 large models (500MB+ for SegFormer)
- Limit: Embedded systems with <2GB RAM cannot load multiple model variants
- Scaling path: Model quantization (int8); model switching via model zoo pattern; lazy-load modules

**CSV logging without circular buffer:**
- Current capacity: ~1 hour of 8 FPS = 28,800 rows before memory issues
- Limit: Pandas DataFrame grows unbounded; no disk flushing
- Scaling path: Implement HDF5 rolling buffer; async write to disk; configurable retention

## Dependencies at Risk

**Deprecated transformers library import pattern:**
- Risk: Using `AutoImageProcessor` and `SegformerForSemanticSegmentation` directly; HuggingFace API changes regularly
- Files: `fast_road_detector.py` (lines 13-14, 170-175)
- Impact: Future transformers updates may change these classes
- Migration plan: Lock transformers version; implement wrapper class that abstracts model interface; periodically test against new versions

**YOLOv8 ultralytics dependency:**
- Risk: Optional dependency handled with try/except (graceful fallback); but no version pin in code
- Files: `live_heading_demo.py` (lines 228-241)
- Impact: API changes in ultralytics could break object detection silently
- Migration plan: Pin ultralytics version; verify YOLOv8n model hash before inference; provide offline model download script

**PySerial for GPS/Scooter communication:**
- Risk: Windows/Linux differences in serial port naming; no fallback if serial library missing
- Files: `camera_waypoint_pipeline.py` (lines 122-136), `live_heading_demo.py` (lines 356-360)
- Impact: Different behavior on different OSes; GPS integration disabled if dependency missing
- Migration plan: Wrap serial operations in OS-agnostic layer; provide mock serial for testing; document supported platforms

## Missing Critical Features

**No command validation or rate limiting:**
- Problem: Robot accepts any steering angle without checking physical limits or rate constraints
- Blocks: Cannot guarantee hardware safety; no emergency stop mechanism
- Files: `realtime_nav_core.py` (lines 875-876), `live_heading_demo.py` (1470-1490 command output)
- Impact: Scooter could receive unstable commands causing jerky motion or hardware damage
- Fix: Implement rate limiting (max change per cycle); clamp steering to physical limits; add watchdog timeout

**No obstacle avoidance when detection fails:**
- Problem: When YOLOv8 crashes or is disabled, robot has no way to detect humans/bicycles
- Blocks: Cannot operate in populated areas safely
- Files: `live_heading_demo.py` (lines 228-241, no fallback detection method)
- Impact: Safety risk in environments with pedestrians
- Fix: Implement conservative default (assume obstacles everywhere); use depth/thermal fallback; require explicit safe mode

**No automatic map building or memory:**
- Problem: Every run starts fresh; no learning of problematic junctions or dead ends
- Blocks: Cannot improve over time; same confusion at same junctions
- Files: `realtime_nav_core.py` has no persistent state, only frame-to-frame memory
- Impact: Inefficient path planning on second visits; branch oscillation at junctions
- Fix: Save skeleton graphs; reuse good paths; build simple occupancy grid

**No failure mode diagnostics:**
- Problem: When path disappears suddenly, log shows discontinuity but not why (bad segmentation? occlusion?)
- Blocks: Difficult to debug real-world failures
- Files: `realtime_nav_core.py` (lines 784-810 path selection) has no diagnostic flags
- Impact: Post-analysis of logs shows WHAT happened but not root cause
- Fix: Add confidence scores; flag segmentation quality; track path stability over time

## Test Coverage Gaps

**No unit tests for BEV calibration:**
- What's not tested: Perspective transform matrix validation; corner-case calibration points (collinear, outside image, etc.)
- Files: `live_heading_demo.py` (lines 752-810)
- Risk: Invalid calibration silently produces wrong geometry
- Priority: HIGH - geometry errors cascade through entire pipeline

**No integration tests for path extraction end-to-end:**
- What's not tested: Skeleton quality with various mask patterns; graph connectivity; DFS path coverage
- Files: `realtime_nav_core.py` (entire BEVPathExtractor class)
- Risk: Logic errors in graph search may only manifest on specific mask patterns found in real video
- Priority: HIGH - core algorithm has no regression tests

**No stress tests for long-running streams:**
- What's not tested: Memory leaks over 1+ hour runs; temporal filter state consistency; CSV unbounded growth
- Files: `live_heading_demo.py` (entire main loop)
- Risk: Deployment finds memory exhaustion hours into run
- Priority: MEDIUM - affects field deployments

**No tests for edge cases in pure pursuit:**
- What's not tested: Behavior at path endpoints; lookahead exceeding path length; zero-curvature paths; backward motion
- Files: `realtime_nav_core.py` (lines 850-907)
- Risk: Undefined behavior with unusual paths
- Priority: MEDIUM - affects robustness to unusual geometries

**No validation tests for serial commands:**
- What's not tested: Scooter handles speed/steer combinations safely; rate limiting works; out-of-range values rejected
- Files: `live_heading_demo.py` (lines 525-548)
- Risk: Hardware damage from invalid commands
- Priority: HIGH - safety-critical

---

*Concerns audit: 2026-03-04*
