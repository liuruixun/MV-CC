import time
import os
import torch.optim
from torch.utils import data
import argparse
import json
from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset,LEVIRCCDataset_video
from data.Dubai_CC.DubaiCC import DubaiCCDataset
from model.model_encoder import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer,DecoderTransformer_video
from utils import *
from model.model_decoder import DecoderTransformer_video
from model.video_encoder import Video_encoder,Sty_fusion
from tqdm import tqdm
import sys
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
your_library_path = '/data/lky/proj/InterVideo/InternVideo2_Chat_8B_InternLM2_5'
# 添加到sys.path
sys.path.append(your_library_path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    """
    Testing.
    """
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    if os.path.exists(args.savepath)==False:
        os.makedirs(args.savepath)
    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)

    snapshot_full_path = os.path.join(args.savepath, args.checkpoint)
    print(f'loading dict from{snapshot_full_path}')
    checkpoint = torch.load(snapshot_full_path,weights_only=True,map_location='cpu')
    video_encoder=Video_encoder()
    sty_fusion=Sty_fusion()
    decoder = DecoderTransformer_video(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim, vocab_size=len(word_vocab), max_lengths=args.max_length, word_vocab=word_vocab, n_head=args.n_heads,
                    n_layers= args.decoder_n_layers, dropout=args.dropout)
    
    video_encoder.load_state_dict(checkpoint['video_encoder_dict'])
    decoder.load_state_dict(checkpoint['decoder_dict'])
    sty_fusion.load_state_dict(checkpoint['sty_fusion_dict'])
    
    # Move to GPU, if available
    sty_fusion=sty_fusion.cuda()
    sty_fusion.eval()
    video_encoder = video_encoder.cuda()
    video_encoder.eval()
    video_encoder = video_encoder.cuda()
    decoder.eval()
    decoder = decoder.cuda()

    # Custom dataloaders
    if args.data_name == 'LEVIR_CC':
        #LEVIR:
        nochange_list = ["the scene is the same as before ", "there is no difference ",
                         "the two scenes seem identical ", "no change has occurred ",
                         "almost nothing has changed "]
        test_loader = data.DataLoader(
            LEVIRCCDataset_video(args.data_folder, args.list_path, 'test', args.token_folder, args.vocab_file, args.max_length, args.allow_unk,if_mask=True,mask_mode=args.mode),
            batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.data_name == 'Dubai_CC':
        #Dubai:
        nochange_list = ["Nothing has changed ", "There is no difference ", "All remained the same "
                            "Everything remains the same ", "Nothing has changed in this area ", "Nothing changed in this area "
                            "No changes in this area ", "No change was made ", "There is no change to mention ", "No changes to mention "
                            "No changed to mention ", "No difference in this area ", "No change to mention "
                            "No change was made ", "No change occurred in this area ", "The area appears the same "]
        test_loader = data.DataLoader(
            DubaiCCDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    l_resize1 = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
    l_resize2 = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
    # Epochs
    test_start_time = time.time()
    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)
    change_references = list()
    change_hypotheses = list()
    nochange_references = list()
    nochange_hypotheses = list()
    change_acc=0
    nochange_acc=0
    print(f'test begin! {len(test_loader)} pictures to test')
    with torch.no_grad():
        # Batches
        for ind, (video_tensor, token_all, token_all_len, _, _,name, mask) in tqdm(enumerate(test_loader)):

            # Move to GPU, if available
            video_tensor=video_tensor.cuda()
            mask=mask.cuda()
            vedie_emb,_=video_encoder(video_tensor)

            vedie_emb=sty_fusion(vedie_emb,mask)
            token_all = token_all.squeeze(0).cuda()

            # Forward prop.
            
            seq = decoder.sample(vedie_emb, k=1)
            

            img_token = token_all.tolist()
            img_tokens = list(map(lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}],
                    img_token))  # remove <start> and pads
            references.append(img_tokens)

            pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
            hypotheses.append(pred_seq)
            assert len(references) == len(hypotheses)
            
            pred_caption = ""
            ref_caption = ""
            for i in pred_seq:
                pred_caption += (list(word_vocab.keys())[i]) + " "
            ref_caption = ""
            for i in img_tokens[0]:
                ref_caption += (list(word_vocab.keys())[i]) + " "
            ref_captions = ""
            for i in img_tokens:
                for j in i:
                    ref_captions += (list(word_vocab.keys())[j]) + " "
                ref_captions += ".    "

            if ref_caption in nochange_list:
                nochange_references.append(img_tokens)
                nochange_hypotheses.append(pred_seq)
                if pred_caption in nochange_list:
                    nochange_acc = nochange_acc+1
            else:
                change_references.append(img_tokens)
                change_hypotheses.append(pred_seq)
                if pred_caption not in nochange_list:
                    change_acc = change_acc+1


        test_time = time.time() - test_start_time
        # Calculate evaluation scores

        if len(nochange_references)>0:
            print('nochange_metric:')
            nochange_metric = get_eval_score(nochange_references, nochange_hypotheses)
            Bleu_1 = nochange_metric['Bleu_1']
            Bleu_2 = nochange_metric['Bleu_2']
            Bleu_3 = nochange_metric['Bleu_3']
            Bleu_4 = nochange_metric['Bleu_4']
            Meteor = nochange_metric['METEOR']
            Rouge = nochange_metric['ROUGE_L']
            Cider = nochange_metric['CIDEr']
            print('BLEU-1: {0:.4f}\t' 'BLEU-2: {1:.4f}\t' 'BLEU-3: {2:.4f}\t' 
                  'BLEU-4: {3:.4f}\t' 'Rouge: {4:.4f}\t' 'Meteor: {5:.4f}\t' 'Cider: {6:.4f}\t'
                  .format(Bleu_1, Bleu_2, Bleu_3, Bleu_4, Rouge, Meteor, Cider))
            print("nochange_acc:", nochange_acc / len(nochange_references))
        if len(change_references)>0:
            print('change_metric:')
            change_metric = get_eval_score(change_references, change_hypotheses)
            Bleu_1 = change_metric['Bleu_1']
            Bleu_2 = change_metric['Bleu_2']
            Bleu_3 = change_metric['Bleu_3']
            Bleu_4 = change_metric['Bleu_4']
            Meteor = change_metric['METEOR']
            Rouge = change_metric['ROUGE_L']
            Cider = change_metric['CIDEr']
            print('BLEU-1: {0:.4f}\t' 'BLEU-2: {1:.4f}\t' 'BLEU-3: {2:.4f}\t'
                  'BLEU-4: {3:.4f}\t' 'Rouge: {4:.4f}\t' 'Meteor: {5:.4f}\t' 'Cider: {6:.4f}\t'
                  .format(Bleu_1, Bleu_2, Bleu_3, Bleu_4, Rouge, Meteor, Cider))
            print("change_acc:", change_acc / len(change_references))

        score_dict = get_eval_score(references, hypotheses)
        Bleu_1 = score_dict['Bleu_1']
        Bleu_2 = score_dict['Bleu_2']
        Bleu_3 = score_dict['Bleu_3']
        Bleu_4 = score_dict['Bleu_4']
        Meteor = score_dict['METEOR']
        Rouge = score_dict['ROUGE_L']
        Cider = score_dict['CIDEr']
        print('Testing:\n' 'Time: {0:.3f}\t' 'BLEU-1: {1:.4f}\t' 'BLEU-2: {2:.4f}\t' 'BLEU-3: {3:.4f}\t' 
              'BLEU-4: {4:.4f}\t' 'Rouge: {5:.4f}\t' 'Meteor: {6:.4f}\t' 'Cider: {7:.4f}\t'
              .format(test_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Rouge, Meteor, Cider))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Captioning')

    # Data parameters
    parser.add_argument(
        '--data_folder', default='/root/Data/LEVIR-MCI-dataset/images', help='folder with data files')
    parser.add_argument('--list_path', default='./data/LEVIR_CC/', help='path of the data lists')
    parser.add_argument('--token_folder', default='./data/LEVIR_CC/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='path of the data lists')
    parser.add_argument('--data_name', default="LEVIR_CC",help='base name shared by data files.')
    parser.add_argument('--checkpoint', default='MV_CC_label.pth',
                        help='path to checkpoint, None if none.')
    parser.add_argument('--mode', default='label',
                    help='path to checkpoint, None if none.')
    #parser.add_argument('--data_folder', default='/root/Data/Dubai_CC/DubaiCC500impair/datasetDubaiCCPublic/imgs_tiles/RGB/',help='folder with data files')
    #parser.add_argument('--list_path', default='./data/Dubai_CC/', help='path of the data lists')
    #parser.add_argument('--token_folder', default='./data/Dubai_CC/tokens/', help='folder with token files')
    #parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    #parser.add_argument('--max_length', type=int, default=27, help='path of the data lists')
    #parser.add_argument('--allow_unk', type=int, default=0, help='path of the data lists')
    #parser.add_argument('--data_name', default="Dubai_CC",help='base name shared by data files.')
    #parser.add_argument('--checkpoint', default='Dubai_CC_batchsize_32_resnet101.pth', help='path to checkpoint, None if none.')

    parser.add_argument('--network', default='resnet101', help='define the encoder to extract features:resnet101,vgg16')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
    parser.add_argument('--workers', type=int, default=2, help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--encoder_dim',default=2048, help='the dimension of extracted features using different network:2048,512')
    parser.add_argument('--feat_size', default=16, help='define the output size of encoder to extract features')
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--decoder_n_layers', type=int, default=1)

    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--attention_dim', type=int, default=2048)
    parser.add_argument('--feature_dim', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # Test
    parser.add_argument('--test_batchsize', default=1, help='batch_size for validation')
    parser.add_argument('--savepath', default="./models_checkpoint/")
    
    args = parser.parse_args()
    main(args)