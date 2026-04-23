# Final Report Outline

Use this structure so the final report is easy to follow and fully self-contained.

## 1. Problem

- Explain the real-world problem in plain language.
- State the exact goal:
  `Develop a machine learning system to predict Florida program earnings and identify programs that perform above or below expected outcomes using multi-year earnings data and XGBoost.`

## 2. Stakeholder and Use Case

- Explain who would use this work.
- Good examples:
  - students comparing programs
  - schools reviewing program outcomes
  - instructors grading the project as a full data science workflow

## 3. Data Sources

- College Scorecard program-level earnings and school features
- Florida-focused cleaned dataset used for training and scoring
- Saved multi-year residual file used for the above/below expected analysis

## 4. Preprocessing

- Explain the main cleaning steps in a short list.
- Mention missing values, feature selection, and the 1-, 4-, and 5-year targets.

## 5. EDA Highlights

- Add the main patterns you found in the data.
- Keep this focused on what helped the modeling decisions.

## 6. Modeling

- Compare the models you tried.
- Explain why XGBoost was selected for deployment.
- Include one simple results table for 1-year, 4-year, and 5-year performance.

## 7. Model Interpretation

- Include the SHAP summary plot for the deployed model.
- Write a short explanation of the top features.

## 8. Deployment

- Explain the FastAPI prediction endpoints.
- Explain the Streamlit dashboard.
- Mention individual prediction, batch prediction, and the multi-year comparison view.

## 9. Assumptions and Limitations

- Florida-only scope
- Program-at-school level, not overall school quality
- Earnings are for graduates working and not enrolled
- Missing data and coverage limits
- Above or below expected is relative to the model, not proof of causation

## 10. Conclusion

- Summarize what the system can do now.
- End with one or two realistic next steps.
