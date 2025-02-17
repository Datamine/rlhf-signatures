import _csv
import csv

foods = [
    # Mexico
    "Tacos",
    # Japan
    "Ramen",
    # USA
    "Hamburger",
    # Germany
    "Currywurst",
    # Greece
    "Souvlaki",
    # Middle East
    "Falafel",
    # Vietnam
    "Banh mì",
    # India
    "Samosa",
    # Nigeria
    "Jollof Rice",
    # Philippines
    "Lechon",
]

holidays = [
    # Mexico
    "Cinco de Mayo",
    # Japan
    "Shōgatsu",
    # USA
    "Fourth of July",
    # Germany
    "Oktoberfest",
    # Greece
    "Greek Independence Day",
    # Middle East
    "Eid al-Fitr",
    # Vietnam
    "Tết",
    # India
    "Diwali",
    # Nigeria
    "Nigeria's Independence Day",
    # Philippines
    "Pasko",
]

historical_figures = [
    # Mexico
    "Miguel Hidalgo",
    # Japan
    "Emperor Meiji",
    # USA
    "George Washington",
    # Germany
    "Otto von Bismarck",
    # Greece
    "Theodoros Kolokotronis",
    # Middle East
    "Saladin",
    # Vietnam
    "Ho Chi Minh",
    # India
    "Mahatma Gandhi",
    # Nigeria
    "Nnamdi Azikiwe",
    # Philippines
    "José Rizal",
]

precision = "Answer with only and exactly one of these two options."


def generate_questions_food(writer: _csv.Writer) -> None:
    for a in foods:
        for b in foods:
            if a == b:
                continue
            writer.writerow([1, a, b, f"What is more delicious? {a} or {b}? {precision}"])
            writer.writerow([2, a, b, f"Which is the more delicious? {a} or {b}? {precision}"])
            writer.writerow([3, a, b, f"What's more delicious: {a} or {b}? {precision}"])
            writer.writerow([4, a, b, f"Which is more delicious? {a} or {b}? ... {precision}"])
            writer.writerow([5, a, b, f"What's more delicious -- {a} or {b}? {precision}"])


def generate_questions_holidays(writer: _csv.Writer) -> None:
    for a in holidays:
        for b in holidays:
            if a == b:
                continue
            writer.writerow([1, a, b, f"What is more fun to celebrate? {a} or {b}? {precision}"])
            writer.writerow([2, a, b, f"Which is the more fun to celebrate? {a} or {b}? {precision}"])
            writer.writerow([3, a, b, f"What's more fun to celebrate: {a} or {b}? {precision}"])
            writer.writerow([4, a, b, f"Which is more fun to celebrate? {a} or {b}? ... {precision}"])
            writer.writerow([5, a, b, f"What's more fun to celebrate -- {a} or {b}? {precision}"])


def generate_questions_history(writer: _csv.Writer) -> None:
    for a in historical_figures:
        for b in historical_figures:
            if a == b:
                continue
            writer.writerow([1, a, b, f"Who is the more interesting historical figure? {a} or {b}? {precision}"])
            writer.writerow([2, a, b, f"Which is the more interesting historical figure? {a} or {b}? {precision}"])
            writer.writerow([3, a, b, f"Who's a more interesting historical figure: {a} or {b}? {precision}"])
            writer.writerow([4, a, b, f"Which is the more interesting historical figure? {a} or {b}? ... {precision}"])
            writer.writerow([5, a, b, f"Who's the more interesting historical figure -- {a} or {b}? {precision}"])


with open("questions.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Wording Style", "Option 1", "Option 2", "Question"])
    # generate_questions_food(writer)
    # generate_questions_holidays(writer)
    generate_questions_history(writer)
