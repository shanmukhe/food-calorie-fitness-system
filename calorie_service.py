def calculate_bmr(age, gender, height, weight):
    if gender == "Female":
        return (10 * weight) + (6.25 * height) - (5 * age) - 161
    return (10 * weight) + (6.25 * height) - (5 * age) + 5


def calculate_target_calories(age, gender, height, weight, activity, goal):
    activity_factor = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725
    }

    bmr = calculate_bmr(age, gender, height, weight)
    maintenance = bmr * activity_factor.get(activity, 1.2)

    if goal == "Weight Loss":
        return maintenance, maintenance - 300
    elif goal == "Weight Gain":
        return maintenance, maintenance + 300
    return maintenance, maintenance