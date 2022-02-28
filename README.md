# KoreanNLI

## Summary

두 문장간의 관계를 통해 논리적으로 타당한지 여부를 예측하는 NLI task를 수행하는 공모전이었다.  Pre-trained 모델 사용, Augmentation, Ensemble 등을 실험해보았고, RoBERTa-large모델을 사용하여 자체 최고 성능을 달성하였다.

## Data

- dataset은 KNLI를 목적으로 생성되었기 때문에 두개의 text(premise, hypothesis)와 두문장의 관계를 나타내는 label(entailment, contradiction, neutral)로 이루어져있다.
    - **예시**
        - premise: 씨름은 상고시대로부터 전해져 내려오는 남자들의 대표적인 놀이로서, 소년이나 장정들이 넓고 평평한 백사장이나 마당에서 모여 서로 힘과 슬기를 겨루는 것이다.
        - hypothesis: 씨름의 여자들의 놀이이다.
        - label: contradiction
- 위의 예시처럼 두개의 text를 이해하고 관계를 예측하는 것이 본 공모전의 과제이다.

## Preprocessing

### Tokenizer

- 기본적인 BERT모델의 tokenizer를 이용하여 두문장을 하나의 input으로 연결하였다.
- Huggingface에 공개된 roberta-large모델의 Tokenizer를 활용하였다.
- Tokenizer의 결과는 다음과 같다.
    
    ```
    {'input_ids': [0, 24746, 2275, 12465, 2481, 27135, 25, 2377, 2283, 2170, 547, 2113, 2689, 2219, 3606, 2, 24746, 2275, 12465, 2481, 27135, 3633, 2377, 1378, 2249, 2370, 2182, 18, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    ```
    
    - Bert기반의 Model은 위의 Tokenizer결과에 따라 ‘input_ids’, ‘token_type_ids’, ‘attention_mask’를 모두 Input으로 받는다.
    - **Input_ids**
        - Input된 토큰들을 Tokenizer의 단어사전에 따라 숫자로 치환한 결과 값이다.
        - Tokenizer는 두 문장을 이어주고, 문장을 max_length에 맞게 Padding을 해주는 Special Token들을 추가한다.
            
            0(시작 토큰), 1(패딩 토큰), 2(종결 및 연결 토큰) 이 Special Token의 예시이다.
            
    - token_type_ids
        - Fine tuning을 위한 Token들 이므로, 모두 0을 할당한다,
    - attention_mask:
        - Padding인 Token은 0으로, 실제 Input Token은 1로 표시한다.

### Augmentation

- 논문 <Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks>를 바탕으로 Random Swap과 Random Delection을 구현하여 시도하였다.
- Random Swap
    - 문장의 Token들의 위치를 서로 바꾸는 것이다.
- Random Delection
    - 문장의 Token들에 임의의 가중치를 매기고 0.2 이하의 가중치가 나오는 토큰들을 제거하는 방법이다.
- Augmentation이 Validation에서는 좋은 성능을 보였으나 실제 Test set에 대해서는 좋지 못한 결과를 보였다. 오히려 과적합을 유도하나?

## Modeling

### RoBERTa-large

- 여러가지 모델들을 실험했지만 결과적으로 최고의 성능을 낸것은 RoBERTa-large 모델을 fine-tuning한 결과 였다.
- 초기엔 RoBERTa-base를 이용하였지만 성능이 잘 오르지 않아 RoBERTa-large를 이용하였다. 이 때 비약적인 성능 향상을 이룰 수 있었다. (역시 large 모델인가..)

[RoBERTa](https://www.notion.so/RoBERTa-2727e2bc00d74527b6f135d9ca1cdf11)

### Experiments

그동안 실험했던 모델들은 다음과 같다.

- Bert-base
- Albert-base
- koBigBird-base : base모델로는 최고 성능
- RoBERTa-base & KoBigBird Ensemble

마지막으론 KoBigBird와 RoBERTa-large를 앙상블 하려 했지만 GPU의 한계로 극복하지 못함..

## Contribution

NLP과제를 공모전으로 나가본 것은 처음이었다. 저번 Object Detection과 마찬가지로 Baseline code를 많이 참고 했지만 그래도 독자적인 길을 가보려 Augentation등을 시도해보았다. 하지만 결국 모델의 용량차이라는 결론을 얻으며 마무리했다.

아직 대회가 마무리되지 않아 다른 사람의 Code를 보진 못했지만 결국 large모델을 쓴 결과 일 것이다. Colab의 GPU에 한계가 있어 더 많은 실험과 지속적인 실험을 하지 못한 것이 아쉽다. 

군대 들어가기 전 마지막 공모전이었다. 단순 ML공모전 부터 해서 DL과 ML을 모두 섞어쓰는 공모전도 해보고, 아이디어 공모전도 해보고 결국 NLP와 CV공모전까지 혼자 할 수 있게 되어 스스로 많이 발전한 것 같다. 이제 들어가서 어떻게 해서 이 공부의 지속성을 이어나갈 수 있을 것인지 고민해보고 준비해야 할 것 같다.
