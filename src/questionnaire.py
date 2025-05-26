import json
import os

def load_questionnaire():
    questions = [
        "Over the last 2 weeks, how often have you felt down, depressed, or hopeless?",
        "Over the last 2 weeks, how often have you had little interest or pleasure in doing things?",
        "Over the last 2 weeks, how often have you felt nervous, anxious or on edge?",
        "Over the last 2 weeks, how often have you been unable to stop or control worrying?",
        "Over the last 2 weeks, how often have you had trouble sleeping?",
        "Over the last 2 weeks, how often have you felt tired or had little energy?",
        "Over the last 2 weeks, how often have you had poor appetite or overeating?"
    ]

    scale = {
        "0": "Not at all",
        "1": "Several days",
        "2": "More than half the days",
        "3": "Nearly every day"
    }

    return questions, scale

def run_questionnaire():
    questions, scale = load_questionnaire()
    print("\nMental Health Self-Assessment (PHQ-9/GAD-7 inspired)\n")
    responses = []

    for i, q in enumerate(questions):
        print(f"Q{i+1}: {q}")
        for key, value in scale.items():
            print(f"   {key}: {value}")
        while True:
            answer = input("Your answer (0-3): ")
            if answer in scale:
                responses.append(int(answer))
                break
            else:
                print("Invalid input. Please enter a number between 0 and 3.")

    score = sum(responses)
    result = ""
    if score < 5:
        result = "Minimal risk"
    elif score < 10:
        result = "Mild risk"
    elif score < 15:
        result = "Moderate risk"
    else:
        result = "High risk"

    summary = {
        "responses": responses,
        "total_score": score,
        "risk_level": result
    }

    print("\nAssessment complete.")
    print(f"Total Score: {score}")
    print(f"Mental Health Risk Level: {result}")

    save_path = os.path.join("data", "questionnaire_results.json")
    os.makedirs("data", exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=4)
        print(f"Results saved to {save_path}\n")

if __name__ == '__main__':
    run_questionnaire()
