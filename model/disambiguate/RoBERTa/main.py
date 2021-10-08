import json
import gzip
import pickle
from util import *
from transformers import RobertaTokenizerFast
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification,RobertaForMaskedLM,RobertaClassificationHead
import itertools
import warnings
warnings.filterwarnings('ignore')


class RoBERTa_MLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.classifier = RobertaClassificationHead(config)
        print("\nRoBERTa_MLM")
        self.init_weights()

    def forward(self,\
        args=None,\
        input_ids=None,\
        attention_mask=None,\
        masked_lm_labels=None,\
        labels = None,
        ):

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.lm_head(sequence_output)
        logits = self.classifier(sequence_output)

        total_loss = 0
 
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            total_loss += loss

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            total_loss += args.mlm_loss * masked_lm_loss

        return (total_loss,logits)

MODEL_CLASSES = {
    'fine' : RobertaForSequenceClassification,
    'post' : RobertaForMaskedLM,
    'multi' : RoBERTa_MLM,
}


def _post_step(outputs):
    pass

def train(args = None, model = None, train_dataloader = None, dev_dataloader = None, train_masker = None, eval_masker = None):

    t_total = len(train_dataloader)//args.gradient_accumulation_steps * args.epochs
    no_decay = {"bias","LayerNorm.weight"}
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(params = optimizer_grouped_parameters, lr = args.learning_rate, eps = args.adam_epsilon)    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(t_total * args.warmup_proportion), num_training_steps = t_total)    

    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^14} | {'Elapsed':^9}")
    print("-"*60)

    save_ = open("./{}dev_set_result.txt".format(args.seed), mode="w", encoding="UTF-8-sig", errors="ignore")
    save_.write("Epoch\t\tTrain_loss\t\tVal_loss\t\tACC\n")
    save_.close()

    model.zero_grad()            
    best_loss = float('inf')
    best_acc = 0
    best_model = None
    dev_acc = list()
    dev_loss = list()
    for epoch_i in trange(args.epochs, leave=False):
        t0_epoch = time.time()
        model.train()            

        total_loss = 0


        for step, batch in enumerate(tqdm(train_dataloader, leave=False)):        
            inputs = convert_batch_to_inputs(args=args, batch=batch, masker=train_masker)

            outputs = model(**inputs)

            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            total_loss += loss.detach().cpu().numpy().item()        
            loss.backward()
            del loss
            if not args.mode == 'fine': _post_step(outputs)

            # if args.mode == 'post' or args.mode== 'fine' and args.mlm: _post_step(outputs)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()                            
                optimizer.zero_grad()                    
                torch.cuda.empty_cache()

        train_loss = total_loss / len(train_dataloader)         

        if dev_dataloader is not None: 

            time_elapsed = time.time() - t0_epoch

            loss, acc =  evaluate(args=args, model=model, dev_dataloader=dev_dataloader, eval_masker=eval_masker)
            if best_acc < acc or (best_acc == acc and best_loss < loss):                
                best_epoch = epoch_i + 1
                best_model = copy.deepcopy(model)
                best_acc = acc
                best_loss = loss
                save_checkpoint(args, best_model)

            print(f"{epoch_i + 1:^7} | {train_loss:^12.6f} | {loss:^10.6f} | {acc:^14.5f} | {time_elapsed:^9.2f}")
            save_ = open("./{}dev_set_result.txt".format(args.seed), mode="a", encoding="UTF-8-sig", errors="ignore")
            save_.write("{0}\t\t{1}\t\t{2}\t\t{3}\n".format(epoch_i,round(train_loss,5),loss,acc))
            dev_acc.append(str(acc))
            dev_loss.append(str(loss))
            save_.close()

    print("\nTraining complete!")
    print(f"Best model epoch: {best_epoch}") 
    save_ = open("./{}dev_set_result.txt".format(args.seed), mode="a", encoding="UTF-8-sig", errors="ignore")
    acc = ("\t").join(dev_acc) + "\n"
    loss = ("\t").join(dev_loss) + "\n"
    save_.write("Dev Acc : {}".format(acc))
    save_.write("Dev Loss : {}".format(loss))
    save_.close()

    return best_model

def evaluate(args=None, model=None, dev_dataloader=None, eval_masker=None, test=False):
    model.eval()      
    preds_list = list()
    labels_list = list()    
    val_accuracy = 0    
    eval_loss = 0       

    n_true = 0
    n_pred = 0
    n_correct = 0

    for step, batch in enumerate(tqdm(dev_dataloader, desc="Evaluate : ", leave=False)):

        inputs = convert_batch_to_inputs(args=args, batch=batch, masker=eval_masker)

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs[0]
            eval_loss += loss.mean().item()

        if not args.mode == 'post':
            preds = torch.argmax(outputs[1], dim=1).flatten()
            preds_list.extend(p.item() for p in preds)            
            labels_list.extend(l.item() for l in inputs['labels'])
            accuracy = accuracy_score(labels_list, preds_list) * 100        
        del inputs

    eval_loss = eval_loss/(step+1)
    if test:
        if not args.mode == 'post':
            print("\nVAL TEST ACC : {}".format(accuracy))
            print("\naccuracy_score \t:", accuracy)
            print("Precision \t:", precision_score(labels_list, preds_list, average='macro') * 100)
            print("Recall \t:", recall_score(labels_list, preds_list, average='macro') * 100)
            print("F1 \t:", f1_score(labels_list, preds_list, average='macro') * 100, "\n")

            save_ = open("./{}test_set_result.txt".format(args.seed), mode="w", encoding="UTF-8-sig", errors="ignore")
            save_.write("Accuracy\t:\t" + str(accuracy_score(labels_list, preds_list) * 100) + "\n")
            save_.write("Precision\t:\t" + str(precision_score(labels_list, preds_list, average='macro') * 100) + "\n")
            save_.write("Recall\t:\t" + str(recall_score(labels_list, preds_list, average='macro') * 100) + "\n")            
            save_.write("F1 Score\t:\t" + str(f1_score(labels_list, preds_list, average='macro') * 100) + "\n")                

            save_.close()
        elif args.mode == 'post':
            perplexity = torch.exp(torch.tensor(eval_loss))

            print("\nPerplexity: {}".format(perplexity))

            save_ = open("./{}test_set_result.txt".format(args.seed), mode="w", encoding="UTF-8-sig", errors="ignore")
            save_.write("Perplexity\t:\t" + str(perplexity) + "\n")

    if args.mode == 'post':
        perplexity = torch.exp(torch.tensor(eval_loss))
        accuracy = perplexity


    return eval_loss, accuracy


if __name__ == "__main__":
    # argparser 선언
    parser = default_parser(argparse.ArgumentParser())
    args = parser.parse_args()

    # Seed 및 cuda 선언
    seed_everything(args.seed)
    set_device(args)


    tokenizer = RobertaTokenizerFast.from_pretrained(args.model)
    special_tokens_dict = {'additional_special_tokens': ['<USER>','<SYS>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # print("\n{}개 신규 토큰 추가".format(num_added_toks))
    config = AutoConfig.from_pretrained(args.model)
    config.num_labels = 2


    # 데이터 불러오기
    tokenized_train, label_train = tokenize_split_history_and_current(args, args.train_file, tokenizer)
    tokenized_dev, label_dev   = tokenize_split_history_and_current(args, args.dev_file, tokenizer)
    tokenized_test, label_test   = tokenize_split_history_and_current(args, args.devtest_file, tokenizer)

    print("\n========== [ Dataset ] ==========")
    print("Train : {}".format(len(tokenized_train['input_ids'])))
    print("Dev : {}".format(len(tokenized_dev['input_ids'])))
    print("Test : {}".format(len(tokenized_test['input_ids'])))

    train_dataloader = idx_to_tensor(args, tokenized_train, label_train, train=True)
    dev_dataloader = idx_to_tensor(args, tokenized_dev, label_dev, train=False)
    test_dataloader = idx_to_tensor(args, tokenized_test, label_test, train=False)

    model = MODEL_CLASSES[args.mode].from_pretrained(pretrained_model_name_or_path=args.model, config=config)
    model.resize_token_embeddings(len(tokenizer))
    if args.model_name_or_path:
        print("Load trained Model : {}".format(args.model_name_or_path))
        args.model_name_or_path = "{}/best_model.bin".format(args.model_name_or_path)
        model = MODEL_CLASSES[args.mode].from_pretrained(pretrained_model_name_or_path=args.model_name_or_path, config=config)                
    model.to(args.device)        

    train_masker = get_masker(args, tokenizer)
    eval_masker = get_masker(args, tokenizer)

    if args.train:
        model = train(args, model, train_dataloader, dev_dataloader, train_masker, eval_masker)

    if args.eval:
        print("\n\n******** Test *********")
        evaluate(args=args, model=model, dev_dataloader=test_dataloader, eval_masker=eval_masker, test=True)