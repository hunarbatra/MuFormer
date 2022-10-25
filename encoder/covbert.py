from transformers import AutoTokenizer, AutoModelForMaskedLM

import esm
import gc
import torch
import numpy as np

import requests

pdb_id = "7LWV" 

data = requests.get(f'https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/{pdb_id}').json()[pdb_id.lower()]
sequence = data[0]['sequence']

tokenizer = AutoTokenizer.from_pretrained("hunarbatra/CoVBERT")
model = AutoModelForMaskedLM.from_pretrained("hunarbatra/CoVBERT")
esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()

# CoVBERT all 261 chars dict:
covbert_full_vocab = bpe_tokenizer = {"<s>":0,"<pad>":1,"</s>":2,"<unk>":3,"<mask>":4,"!":5,"\"":6,"#":7,"$":8,"%":9,"&":10,"'":11,"(":12,")":13,"*":14,"+":15,",":16,"-":17,".":18,"/":19,"0":20,"1":21,"2":22,"3":23,"4":24,"5":25,"6":26,"7":27,"8":28,"9":29,":":30,";":31,"<":32,"=":33,">":34,"?":35,"@":36,"A":37,"B":38,"C":39,"D":40,"E":41,"F":42,"G":43,"H":44,"I":45,"J":46,"K":47,"L":48,"M":49,"N":50,"O":51,"P":52,"Q":53,"R":54,"S":55,"T":56,"U":57,"V":58,"W":59,"X":60,"Y":61,"Z":62,"[":63,"\\":64,"]":65,"^":66,"_":67,"`":68,"a":69,"b":70,"c":71,"d":72,"e":73,"f":74,"g":75,"h":76,"i":77,"j":78,"k":79,"l":80,"m":81,"n":82,"o":83,"p":84,"q":85,"r":86,"s":87,"t":88,"u":89,"v":90,"w":91,"x":92,"y":93,"z":94,"{":95,"|":96,"}":97,"~":98,"¡":99,"¢":100,"£":101,"¤":102,"¥":103,"¦":104,"§":105,"¨":106,"©":107,"ª":108,"«":109,"¬":110,"®":111,"¯":112,"°":113,"±":114,"²":115,"³":116,"´":117,"µ":118,"¶":119,"·":120,"¸":121,"¹":122,"º":123,"»":124,"¼":125,"½":126,"¾":127,"¿":128,"À":129,"Á":130,"Â":131,"Ã":132,"Ä":133,"Å":134,"Æ":135,"Ç":136,"È":137,"É":138,"Ê":139,"Ë":140,"Ì":141,"Í":142,"Î":143,"Ï":144,"Ð":145,"Ñ":146,"Ò":147,"Ó":148,"Ô":149,"Õ":150,"Ö":151,"×":152,"Ø":153,"Ù":154,"Ú":155,"Û":156,"Ü":157,"Ý":158,"Þ":159,"ß":160,"à":161,"á":162,"â":163,"ã":164,"ä":165,"å":166,"æ":167,"ç":168,"è":169,"é":170,"ê":171,"ë":172,"ì":173,"í":174,"î":175,"ï":176,"ð":177,"ñ":178,"ò":179,"ó":180,"ô":181,"õ":182,"ö":183,"÷":184,"ø":185,"ù":186,"ú":187,"û":188,"ü":189,"ý":190,"þ":191,"ÿ":192,"Ā":193,"ā":194,"Ă":195,"ă":196,"Ą":197,"ą":198,"Ć":199,"ć":200,"Ĉ":201,"ĉ":202,"Ċ":203,"ċ":204,"Č":205,"č":206,"Ď":207,"ď":208,"Đ":209,"đ":210,"Ē":211,"ē":212,"Ĕ":213,"ĕ":214,"Ė":215,"ė":216,"Ę":217,"ę":218,"Ě":219,"ě":220,"Ĝ":221,"ĝ":222,"Ğ":223,"ğ":224,"Ġ":225,"ġ":226,"Ģ":227,"ģ":228,"Ĥ":229,"ĥ":230,"Ħ":231,"ħ":232,"Ĩ":233,"ĩ":234,"Ī":235,"ī":236,"Ĭ":237,"ĭ":238,"Į":239,"į":240,"İ":241,"ı":242,"Ĳ":243,"ĳ":244,"Ĵ":245,"ĵ":246,"Ķ":247,"ķ":248,"ĸ":249,"Ĺ":250,"ĺ":251,"Ļ":252,"ļ":253,"Ľ":254,"ľ":255,"Ŀ":256,"ŀ":257,"Ł":258,"ł":259,"Ń":260}

# CoVBERT order: A C D E F G H I K L M N P Q R S T V W Y
covbert_valid_order = 'A C D E F G H I K L M N P Q R S T V W Y'
covbert_valid_order_list = covbert_valid_order.split(' ')
covbert_valid_order_dict = {a:n for n,a in enumerate(covbert_valid_order_list)}

# CoVBERT order to Original AA Alphabets ordering:
tmp_aa_map = np.array([covbert_valid_order_dict[a] for a in "ARNDCQEGHILKMFPSTWYV"])

# map ESMs encoding to CoVBERTs encoding ->
esm_order_dict = {a:n for n,a in enumerate(alphabet.all_toks[4:24])}

esm_to_covbert = {y+4:covbert_full_vocab[x] for x,y in esm_order_dict.items()}

# change third dim from 261 to 20 AAs
def compute_final_model_out(x, ln):
    # generate models output
    model_out = model(x)["logits"][:,1:(ln+1),37:62]

    final_model_out = torch.zeros(model_out.shape[0], model_out.shape[1], 20)

    for j in range(0, model_out.shape[0]):
        for i in range(0, model_out.shape[1]):
            test = model_out[j][i]
            final_model_out[j][i] = torch.cat([test[:1], test[2:9], test[10:14], test[15:20], test[21:23], test[24:]])  # select all valid AA rows

    return final_model_out

def generate_covbert_bias(seq):
    # input x encoded with ESM vocab tokenizer
    # ln is the length of x
    x, ln = alphabet.get_batch_converter()([(None,seq)])[-1],len(seq)

    # convert x to CoVBERTs encoding scheme
    for i in range(1, x.shape[-1]-1):
        curr_esm_encoding = x[0][i].item()
        covbert_encoding_conversion = esm_to_covbert[curr_esm_encoding]
        x[0][i] = covbert_encoding_conversion
    
    p = ln

    with torch.no_grad():
        # use covberts encoded x now
        # compute model out
        f = lambda x: compute_final_model_out(x, ln)

        logits = np.zeros((ln,20))
        
        # mask different positions
        for n in range(0,ln,p):
            m = min(n+p,ln)
            x_h = torch.tile(torch.clone(x),[m-n,1])
            for i in range(m-n):
                x_h[i,n+i+1] = tokenizer.mask_token_id
            fx_h = f(x_h)
            for i in range(m-n):
                logits[n+i] = fx_h[i,n+i].numpy()

        return logits[:,tmp_aa_map]
    
generate_covbert_bias(sequence)

bias = get_bias_from_esm(sequence)
np.savetxt("covbert-bias.txt", bias)

plt.rcParams["figure.figsize"] = (50,30)
plt.imshow(bias.T)
