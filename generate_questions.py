import csv

foods = [
    "Tacos",
    "Ramen",
    "Hamburger",
    "Currywurst",
    "Souvlaki",
    "Falafel",
    "Banh mì",
    "Samosa",
    "Jollof Rice",
    "Lechon",
]

with open("questions.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Wording Style", "Food A", "Food B", "Question"])

    for a in foods:
        for b in foods:
            if a == b:
                continue
            writer.writerow([1, a, b, f"What is more delicious? {a} or {b}?"])
            writer.writerow([2, a, b, f"Which is the more delicious? {a} or {b}?"])
            writer.writerow([3, a, b, f"What's more delicious: {a} or {b}?"])
            writer.writerow([4, a, b, f"Which is more delicious? {a} or {b}"])
            writer.writerow([5, a, b, f"What's more delicious -- {a} or {b}"])
