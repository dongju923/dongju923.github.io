---
title: "커스텀 dual encoder BERT로 문장 유사도를 계산하기"
categories: Project
toc: true
toc_sticky: true
---

본 포스팅은 지금 하고 있는 연구가 Sentence_transformers 라이브러리의 모델에서 cross-attention을 사용할 수 없어서 transformers 라이브러리의 모델로 변환하는 과정을
담고 있습니다. 도움이 되셨으면 좋겠습니다. ㅎㅎ..

### 모델과 토크나이저 로드
```python
q_encoder = AutoModel.from_pretrained("klue/roberta-base")
p_encoder = AutoModel.from_pretrained("klue/roberta-base")
```
q_encoder와 p_encoder를 선언한다.

```python
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
```
토크나이저도 추가한다.

### 데이터셋 선언
```python
sts_train_dataset = sts_processing("../data/kor_nli_sts/KorSTS/sts-train.tsv")
sts_valid_dataset = sts_processing("../data/kor_nli_sts/KorSTS/sts-dev.tsv")
sts_test_dataset = sts_processing("../data/kor_nli_sts/KorSTS/sts-test.tsv")
```
데이터셋을 추가한다. 데이터셋은 [여기](https://github.com/kakaobrain/kor-nlu-datasets) 에서 받을 수 있다.  
sts_processing 함수는 그냥 NAN 값을 제거하고 score 부분을 0~1로 정규화 하여 Dataset 객체로 반환한다.

### 커스텀 모델 선언
```python
model = DualEncoderModel(q_encoder, p_encoder, tokenizer)
```
모델 부분이다. 허깅페이스에서는 듀얼 인코더 모델이 없는 것 같아서 만들어 보았다. 모델 호출은 q_encoder와
p_encoder, tokenizer를 필요로 한다.

```python
class HuggingfaceBaseDualEncoderModel(nn.Module):
    def __init__(self, q_encoder, p_encoder, tokenizer):
        super(HuggingfaceBaseDualEncoderModel, self).__init__()
        self.q_encoder = q_encoder
        self.p_encoder = p_encoder
        self.pooler = Pooling(word_embedding_dimension=768)
        self.tokenizer = tokenizer

    def forward(self, sen1_features=None, sen2_features=None, labels=None):
        pass
```
모델은 일단 torch.nn.Module을 상속받아서 Base 모델을 선언하였다. 굳이 이렇게 하지 않아도 되지만 현재 하고 있는 연구가
Base 모델 클래스를 기본으로 파생해서 만들어야 하기 때문이다. 
```python
class DualEncoderModel(HuggingfaceBaseDualEncoderModel):
    def forward(self, sen1_features=None, sen2_features=None, labels=None):
        q_embeddings = self.q_encoder(**sen1_features)
        q_embeddings = self.pooler(q_embeddings, sen1_features)
        p_embeddings = self.p_encoder(**sen2_features)
        p_embeddings = self.pooler(p_embeddings, sen2_features)

        # loss
        a = torch.nn.functional.normalize(q_embeddings["sentence_embedding"], p=2, dim=1)
        b = torch.nn.functional.normalize(p_embeddings["sentence_embedding"], p=2, dim=1)
        dot_score = (a * b).sum(dim=-1)
        scores = dot_score * 20
        scores = scores[:, None] - scores[None, :]

        # label matrix indicating which pairs are relevant
        labels = labels[:, None] < labels[None, :]
        labels = labels.float()

        # mask out irrelevant pairs so they are negligible after exp()
        scores = scores - (1 - labels) * 1e12

        # append a zero as e^0 = 1
        scores = torch.cat((torch.zeros(1).to(scores.device), scores.view(-1)), dim=0)
        # scores에서 labels가 0인 경우 log를 취했을 때 nan값이 됨. 따라서 labels가 1인 경우의 score에만 log를 취해서 전부 더함
        loss = torch.logsumexp(scores, dim=0)

        output = {"loss": loss}
        return output
```
코드가 좀 지저분 하긴 하지만, loss 클래스를 따로 만들지 않고 Model에 만들었다. loss는 CosENTLoss()를 사용하였다.    
기본적으로 q_encoder, p_encoder 모델은 RoBERTa 모델이고 forward의 input으로 "input_ids", "attention_mask", "token_type_ids"
를 받는다. sen1_features, sen2_features는 딕셔너리 형태로 되어 있고, 이따 아래에서 설명할 data collator에 의해 생성된다.  
pooler 부분은 sentence_transformer 라이브러리에서 지원하는 Pooling 클래스 모듈이다. sentence_transformer 라이브러리를 안쓴 이유는 
추후 연구 진행에 막힘이 있을 것 같아서 transformer 라이브러리를 직접적으로 사용하였다. `sentence_transformers.models.Pooling()`을 오버라이딩 하여 forward 부분을 
아래처럼 수정하였다.
```python
# 기본 Pooling의 forward
def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
    token_embeddings = features["token_embeddings"]

    
# 수정한 forward
def forward(self, outputs: BaseModelOutputWithPoolingAndCrossAttentions, features: dict[str, Tensor]) -> dict[str, Tensor]:
    features["token_embeddings"] = outputs["last_hidden_state"]
    token_embeddings = features["token_embeddings"]
```

### TrainingArguments 설정
```python
training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        optim='adamw_hf',
        fp16=True,
        bf16=False,
        eval_strategy="epoch",
        eval_steps=100,
        save_strategy="epoch",
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        weight_decay=0.1,
        lr_scheduler_type="linear",
        max_grad_norm=1,
        remove_unused_columns=False,
        **(
            {
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False
            } if early_stopping else {}
        )
    )
```
Trainer에 필요한 args를 설정한다. 사전에 early_stopping 이라는 bool 변수를 두고 True 라면 아래 인자가 추가되게 만들었다.  

### Trainer 설정
```python
trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=sts_train_dataset,
        eval_dataset=sts_valid_dataset,
        data_collator=HuggingfaceDataCollator(tokenizer),
        **(
            {
                "callbacks": [EarlyStoppingCallback(early_stopping_patience=2)] if early_stopping else []
            }
        ) if early_stopping else {}
    )
```
원래 기본 data collator를 사용하려고 했으나 의문의(?) 오류가 발생해서 커스텀 data collator를 선언하였다.
```python
@dataclass
class HuggingfaceDataCollator:
    tokenize_fn: Callable

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # Correct tokenize function usage
        sen1_features = self.tokenize_fn([feature["sentence1"] for feature in features], padding=True, truncation=True,
                                         return_tensors="pt")
        sen2_features = self.tokenize_fn([feature["sentence2"] for feature in features], padding=True, truncation=True,
                                         return_tensors="pt")
        labels = torch.tensor([feature["score"] for feature in features])

        # Return formatted inputs for the model
        return {
            "sen1_features": {key: val for key, val in sen1_features.items()},
            "sen2_features": {key: val for key, val in sen2_features.items()},
            "labels": labels,
        }
```
transformers 라이브러리에서 커스텀으로 collator를 설정하면 dataloader를 불러올 때 커스텀한 collator를 적용해서 데이터를 불러온다.
원래 dataloader를 오버라이딩해서 구현하려고 했지만 너무 헷갈려서 그냥 collator를 수정하였다.  
여기서 features는 배치 64만큼의 리스트로 되어있고 하나의 요소는 {"score": ~~, "sentence1": ~~, "sentence2": ~~} 같이 딕셔너리로 구성되어 있다.  
tokenize_fn은 토크나이저 함수이며 토크나이징을 통해 BERT에 맞게 [CLS] 토큰과 [SEP] 토큰을 추가한다.  
sen1_features와 sen2_features는 토크나이징을 통해 "input_ids", "attention_mask", "token_type_ids"가 자동으로 만들어 진다.
return은 마찬가지로 dict 형태로 반환한다.


Trainer 부분이다. 학습 중간에 validation step이 이루어 지지 않아서 Trainer 클래스의 `compute_loss()` 메서드를 오버라이딩하여 아래와 같이 수정하였다.  
```python
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 모델 입력을 분리
        sen1_features = inputs.pop("sen1_features")
        # collator 에서 딕셔너리 형태로 반환
        sen2_features = inputs.pop("sen2_features")
        labels = inputs.pop("labels")

        # 모델에 forward 호출
        outputs = model(
            sen1_features=sen1_features,
            sen2_features=sen2_features,
            labels=labels
        )
        loss = outputs["loss"]  # loss 가져오기
        return (loss, outputs) if return_outputs else loss
```
`compute_loss()` 메서드에서 model은 DualEncoderModel 모델이고 inputs로는 위의 data collator를 통해서 나온 리턴값이 입력으로 들어온다.  
모델을 호출할 때 model의 forward에 맞게 sen1_features, sen2_features, labels로 나누어서 들어간다. sen1_features, sen2_features는 
각각 "input_ids", "attention_mask", "token_type_ids" 가 들어있다.

### Training
```python
trainer.train()
```
`train()`을 통해 학습을 시작한다. H100 1개 기준 5분 걸렸다.

### Evaluating
이제 모델을 평가할 차례이다. 모델 평가는 `sentence_transformers.evaluation.EmbeddingSimilarityEvaluator`를 오버라이딩 해서 사용하였다.  
이 클래스도 sentence_transformer 모델에 특화되어 있어서 그냥 transformers 라이브러리를 사용하는 나에게는 동작하지 않았다.  
따라서 `def __call__()` 부분만 아래와 같이 수정하였다.
```python
class HuggingfaceEmbeddingSimilarityEvaluator(EmbeddingSimilarityEvaluator):
    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        import numpy as np
        from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, \
            paired_manhattan_distances
        from scipy.stats import pearsonr, spearmanr
        import csv
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"
    
        print(f"EmbeddingSimilarityEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:")
    
        model.eval()
        with torch.no_grad():
            q_encoder = CustomEncoder(model=model.q_encoder, pooler=model.pooler, tokenizer=model.tokenizer)
            p_encoder = CustomEncoder(model=model.p_encoder, pooler=model.pooler, tokenizer=model.tokenizer)
            embeddings1 = q_encoder.encode(
                self.sentences1,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
                precision="float32"
    
            )
            embeddings2 = p_encoder.encode(
                self.sentences2,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
                precision="float32"
            )
```
이후 코드는 원래 클래스와 동일하다. 여기서 문제는 encoder 모델이 huggingface AutoModel 이기 때문에 `encoder()` 함수가 없다.  
(sentence_transformers에는 있음) 따라서 sentence_transformer 라이브러리의 `encoder()` 함수를 참고하여 만들었다.
```python
class CustomEncoder:
    def __init__(self, model, pooler, tokenizer):
        self.model = model
        self.pooler = pooler
        self.tokenizer = tokenizer
        self.device = self.model.device
        self.model.to(self.device)

    def encode(self,
               sentences,
               prompt_name=None,
               prompt=None,
               batch_size=32,
               show_progress_bar=True,
               output_value="sentence_embedding",
               precision="float32",
               convert_to_numpy=True,
               convert_to_tensor=False,
               device=None,
               normalize_embeddings=False,
               ):
        import numpy as np
        from tqdm.autonotebook import trange

        self.model.eval()
        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
                sentences, "__len__"
        ):
            sentences = [sentences]
            input_was_string = True
        extra_features = {}
        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            features = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            features = features.data
            features = self._batch_to_device(features, device)
            features.update(extra_features)

            with torch.no_grad():
                out_features = self.model(**features)
                out_features = self.pooler(out_features, features)
                out_features["sentence_embedding"] = self._truncate_embeddings(
                    out_features["sentence_embedding"], None
                )

                if output_value == "token_embeddings":
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features["attention_mask"]):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0: last_mask_id + 1])
                elif output_value is None:  # Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features["sentence_embedding"])):
                        row = {name: out_features[name][sent_idx] for name in out_features}
                        embeddings.append(row)
                else:  # Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            if len(all_embeddings):
                if isinstance(all_embeddings, np.ndarray):
                    all_embeddings = torch.from_numpy(all_embeddings)
                else:
                    all_embeddings = torch.stack(all_embeddings)
            else:
                all_embeddings = torch.Tensor()
        elif convert_to_numpy:
            if not isinstance(all_embeddings, np.ndarray):
                if all_embeddings and all_embeddings[0].dtype == torch.bfloat16:
                    all_embeddings = np.asarray([emb.float().numpy() for emb in all_embeddings])
                else:
                    all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        elif isinstance(all_embeddings, np.ndarray):
            all_embeddings = [torch.from_numpy(embedding) for embedding in all_embeddings]

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def _truncate_embeddings(self, embeddings, truncate_dim):
        return embeddings[..., :truncate_dim]

    def _batch_to_device(self, batch: dict[str, Any], target_device) -> dict[str, Any]:
        from torch import Tensor
        for key in batch:
            if isinstance(batch[key], Tensor):
                batch[key] = batch[key].to(target_device)
        return batch

    def _text_length(self, text) -> int:
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])
```
이 코드에서 중요한것은 tokenizer를 통해서 나온 결과 값을 모델의 forward로 들어가는 형식과 맞춘것이다. 
코드를 조금 더 간결하게 짤 수 있을 것 같은데 실력의 한계다..

### Result

|        Metric         | Pearson | Spearman |
|:------------:|---------------------|:--------------------:|
|  Cosine   | 0.8124762323646662 |  0.8082641269098775  |
|  Euclidean   | 0.8019528231023273 |  0.798158350294268   |
|  Manhattan   | 0.8001670570954755 |  0.7966446194852113  |
|Dot | 0.6849669599345047 |  0.6659398109956867  |