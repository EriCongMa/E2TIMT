from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import os
smooth = SmoothingFunction()

def doc_bleu(ref, pred):
    '''
    Definition for input:
        ref: a list in which the element is a reference sentence
        pred: a list in which the element is a machine predicted sentence
    Definition for output:
        tok_bleu: a folat number which represents the token level bleu
        chr_bleu: a folat number which represents the character level bleu
    '''
    assert len(ref) == len(pred), (
        "Length of ref and pred are not the same!"
    )
    tok_ref = [[new_ref.strip().split()] for new_ref in ref]
    tok_pred = [new_pred.strip().split() for new_pred in pred]
    
    chr_ref = []
    chr_pred = []

    for idx in range(len(ref)):
        ref_line = ''.join(ref[idx].strip().split())
        pred_line = ''.join(pred[idx].strip().split())
        new_ref = [new_ref for new_ref in ref_line]
        new_pred = [new_pred for new_pred in pred_line]
        chr_ref.append([new_ref])
        chr_pred.append(new_pred)
    
    smooth = SmoothingFunction()
    weight_list = [
                    (1,0,0,0),
                    (0,1,0,0),
                    (0,0,1,0),
                    (0,0,0,1),
                    ]
    
    # score_list = []
    avg_tok_bleu = 0
    avg_chr_bleu = 0
    for idx in range(4):
        current_tok_bleu = corpus_bleu(tok_ref, tok_pred, weights = weight_list[idx], smoothing_function=smooth.method1)
        current_chr_bleu = corpus_bleu(chr_ref, chr_pred, weights = weight_list[idx], smoothing_function=smooth.method1)
        
        # This average doesn't consider length penalty
        avg_tok_bleu += 0.25 * current_tok_bleu
        avg_chr_bleu += 0.25 * current_chr_bleu
    
    # This calculation uses (0.25, 0.25, 0.25, 0.25) as weights. Meanwhile, it also considers the length penalty
    tok_bleu = corpus_bleu(tok_ref, tok_pred, smoothing_function=smooth.method1)
    chr_bleu = corpus_bleu(chr_ref, chr_pred, smoothing_function=smooth.method1)
    
    return chr_bleu, tok_bleu
