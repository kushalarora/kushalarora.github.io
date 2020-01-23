---
layout: page 
permalink: /research_statement
title: Research Statement 
---

My Ph.D. research is primarily focused on posing language generation as a sequential decision-making problem. Looked at from this perspective, the language generation problem can be seen as an instance of imitation learning (IL) and maxium likelihood based training as behavior cloning. The shortcomings of behavior cloning—the need for large amounts of data to cover the whole state space to learn a reasonable policy—are well known in the imitation learning community. This is especially true for complex domains with large state spaces. Language is one such domain with complex reward structures, exponentially exploding state space (if contexts are considered to be a state), the hierarchy at phrasal, sentence and discourse level. 

The behavior cloning view of language model training explains the need for a huge amount of data for training reasonable language generation models as well as their inability to generalize systematically. Additionally, as the MLE based approach encourages the model to blindly mimic the data distribution, it lacks incentives to learn the inherent properties of language like grammaticality, compositionality, discourse structure that lies beyond the surface form. This perspective has motivated me to pursue research at the intersection of IL/ batched RL and NLP into building training algorithms that lead to robust language generation systems that generalize systematically, is compositional, and respect the discourse structure of language.
