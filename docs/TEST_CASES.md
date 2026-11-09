# Test Cases

## A. Automated (`python -m unittest discover -s tests -v`) — all passing at delivery
| ID | Area | Test | Expected |
|---|---|---|---|
| TC-01 | Density | Boundary values 0, 8, 8.1, 18.1 PCU | Low, Low, Medium, High |
| TC-02 | Density | 2 cars + 1 bus + 2 motorcycles | 5.5 PCU |
| TC-03 | Signals | Loads N30/E5/S15/W1 | Green order N > S > E ≥ W |
| TC-04 | Signals | One road 100, others 0 | Every road ≥ MIN_GREEN (no starvation) |
| TC-05 | Signals | Equal demand | Equal green (±1 s) |
| TC-06 | Signals | All-zero and 2-road inputs | Valid plan, sums to cycle |
| TC-07 | Signals | 300 random scenarios | Σgreen + yellows = cycle; all within [min,max] |
| TC-08 | Signals | Skewed traffic | Estimated wait reduction > 0 |
| TC-09 | Prediction | Linearly rising load | Trend = Rising, predicted > current |
| TC-10 | Prediction | Periodic load with history | Gradient-Boosting method used, prediction ≥ 0 |
| TC-11 | Prediction | Empty / 1-point input | "no data" / "persistence", no crash |
| TC-12 | Database | CRUD, cascade delete, bad column | Round-trips; ValueError on bad column |
| TC-13 | App | Every page with empty DB; unknown session | 200; 404 JSON |
| TC-14 | Upload | No file / `.exe` / fake `.mp4` | Specific error messages |
| TC-15 | API | Invalid simulator payloads | HTTP 400 with message |
| TC-16 | Pipeline | 4 synthetic videos of different density | All roads *completed*; PCU North > South > East |
| TC-17 | Pipeline | Signal plan from results | Stored, not provisional, North > West, min green respected |
| TC-18 | Reports | 5 chart PNGs, PDF, CSV, all pages | Valid PNG / `%PDF` / CSV content, HTTP 200 |
| TC-19 | Cleanup | Delete session | 200, then overview → 404 |

## B. Manual — real YOLO (run once on your laptop)
| ID | Steps | Expected |
|---|---|---|
| TC-20 | `python scripts\check_setup.py` | All `[ OK ]`; YOLO ready message |
| TC-21 | Upload one real traffic clip as North | Boxes with labels drawn on the stream; counts > 0; progress bar reaches 100 % |
| TC-22 | Upload 4 clips of clearly different density | Density badges differ; busiest road gets the longest green; all ≥ 10 s |
| TC-23 | Compare screen counts with a manual count of 5 random frames | Counts reasonably close (YOLOv8n misses small/occluded vehicles) |
| TC-24 | Signals page → *Start* simulation | Lights cycle; adaptive total queue ends lower than fixed for skewed traffic |
| TC-25 | What-if: North 40 cars, others 0 | North ≈ max green; other roads = min green |
| TC-26 | Reports → PDF and CSV | Files download and open; charts and plan present |
| TC-27 | Upload a text file renamed `x.mp4` | Rejected with “could not be read” |
| TC-28 | Live page, camera 0 → Start → Stop & save | Live boxes; session appears in Reports |
| TC-29 | Live page with webcam disabled/busy | Clear error message, app keeps running |
| TC-30 | Stop server during processing, restart | Road shows *interrupted*; app starts normally |
