# Homework_for_Transformer-Lecture7-
## 7.1 
- The Transformer itself does not possess sequential information, and positional encoding is used for:
  * Distinguish tokens in different positions
  * Enable the model to understand the "digit weights" of numbers (units, tens, etc.)
## 7.2 
- Residual connections can:
  * Mitigate gradient vanishing
  * Maintain the flow of information
  * Enhance the stability of deep models
## 7.3 
- Compared to single-head attention, It can capture different types of relationships
  - In addition, the following aspects may be considered separately (about current position)
  - Carry information
## 7.4
- Addition requires:
  - Long distance dependency (carry)
  - multi-step reasoning
- A model that is too small can lead to
  * Unable to represent complex rules
  * Lack of learning ability
## 7.5 
  Yes, but more complicated. The model may be too small.