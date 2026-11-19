from evaluation import iou, match, evaluate_image, prf

def close(a, b):
    return abs(a - b) < 1e-9

# IoU
assert close(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
assert close(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
assert close(iou([0, 0, 10, 10], [5, 0, 15, 10]), 50 / 150)
assert close(iou([0, 0, 10, 10], [0, 0, 10, 5]), 0.5)

# exactly 0.5 counts as a match (>=)
assert len(match([[0, 0, 10, 5]], [[0, 0, 10, 10]], 0.5)) == 1

# duplicate prediction on one face: one TP, one FP
r = evaluate_image([[0, 0, 10, 10], [1, 0, 11, 10]], [[0, 0, 10, 10]])
assert (r["tp"], r["fp"], r["fn"], r["count_err"]) == (1, 1, 0, 1)

# one miss and one false alarm: count error hides it
gt = [[0, 0, 10, 10], [50, 50, 60, 60]]
pred = [[0, 0, 10, 10], [100, 100, 110, 110]]
r = evaluate_image(pred, gt)
assert (r["tp"], r["fp"], r["fn"], r["count_err"]) == (1, 1, 1, 0)

# empty cases
assert evaluate_image([], gt)["fn"] == 2
assert evaluate_image(pred, [])["fp"] == 2

# metrics
p, rc, f1 = prf(6, 1, 1)
assert close(p, 6 / 7) and close(rc, 6 / 7) and close(f1, 6 / 7)
assert prf(0, 0, 0) == (0.0, 0.0, 0.0)

print("all evaluation tests passed")