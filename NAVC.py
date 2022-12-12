import torch
import torch.nn as nn

# from examples.video_caption.NAVC import Encoder, Decoder, Predictor, Constants
from seq2seq import Seq2Seq
import Encoder
import Decoder
import Predictor
import Constants

class Joint_Representaion_Learner(nn.Module):
    def __init__(self, feats_size, opt):
        super(Joint_Representaion_Learner, self).__init__()
        self.fusion = opt.get('fusion', 'temporal_concat')

        if self.fusion not in ['temporal_concat', 'addition', 'none']:
            raise ValueError('We now only support the fusion type: temporal_concat | addition | none')

        self.norm_list = []
        self.is_bn = (opt.get('norm_type', 'bn').lower() == 'bn')

        if not opt['no_encoder_bn']:
            if self.fusion == 'addition':
                feats_size = [feats_size[0]]
            for i, item in enumerate(feats_size):
                tmp_module = nn.BatchNorm1d(item) if self.is_bn else nn.LayerNorm(item)
                self.norm_list.append(tmp_module)
                self.add_module("%s%d" % ('bn' if self.is_bn else 'ln', i), tmp_module)

    def forward(self, encoder_outputs, encoder_hiddens):
        if not isinstance(encoder_hiddens, list):
            encoder_hiddens = [encoder_hiddens]
        encoder_hiddens = torch.stack(encoder_hiddens, dim=0).mean(0)

        if self.fusion == 'none':
            if isinstance(encoder_outputs, list):
                encoder_outputs = torch.cat(encoder_outputs, dim=1)
            return encoder_outputs, encoder_hiddens

        if not isinstance(encoder_outputs, list):
            encoder_outputs = [encoder_outputs]

        if self.fusion == 'addition':
            encoder_outputs = torch.stack(encoder_outputs, dim=0).mean(0)

        if len(self.norm_list):
            assert len(encoder_outputs) == len(self.norm_list)
            for i in range(len(encoder_outputs)):
                if self.is_bn:
                    batch_size, seq_len, _ = encoder_outputs[i].shape
                    encoder_outputs[i] = self.norm_list[i](
                        encoder_outputs[i].contiguous().view(batch_size * seq_len, -1)).view(batch_size, seq_len, -1)
                else:
                    encoder_outputs[i] = self.norm_list[i](encoder_outputs[i])

        if self.fusion == 'temporal_concat':
            assert isinstance(encoder_outputs, list)
            encoder_outputs = torch.cat(encoder_outputs, dim=1)

        return encoder_outputs, encoder_hiddens


def get_joint_representation_learner(opt):
    if opt.get('no_joint_representation_learner', False):
        return None
    feats_size = [opt['dim_hidden']] * len(opt['modality'])
    return Joint_Representaion_Learner(feats_size, opt)


def get_auxiliary_task_predictor(opt):
    supported_auxiliary_tasks = [item[10:] for item in dir(Predictor) if 'Predictor_' in item]

    layers = []
    for crit_name in opt['crit']:
        if crit_name in supported_auxiliary_tasks:
            predictor_name = 'Predictor_%s'%crit_name
            _func = getattr(Predictor, predictor_name, None)
            if _func is None:
                raise ValueError('We can not find {} in models/Predictor.py'.format(predictor_name))
            layers.append(_func(opt, key_name=Constants.mapping[crit_name][0]))
    return None if not len(layers) else Predictor.Auxiliary_Task_Predictor(layers)


def print_info(module_name, supported_modules, key_name):
    print('Supported {}:'.format(key_name))
    for item in supported_modules:
        print('- {}{}'.format(item, '*' if module_name == item else ''))

    if module_name not in supported_modules:
        raise ValueError('We can not find {} in models/{}.py'.format(module_name, key_name))


def get_encoder(opt, input_size):
    print_info(
        module_name=opt['encoder'],
        supported_modules=Encoder.__all__,
        key_name='Encoder'
    )
    return getattr(Encoder, opt['encoder'], None)(opt)


def get_decoder(opt):
    print_info(
        module_name=opt['decoder'],
        supported_modules=Decoder.__all__,
        key_name='Decoder'
    )
    return getattr(Decoder, opt['decoder'], None)(opt)


def get_model(opt):
    modality = opt['modality'].lower()
    input_size = []
    mapping = {
        'i': opt['dim_i'],
        'm': opt['dim_m'],
        'a': opt['dim_a'],
        'o': opt['dim_o'],
    }
    for char in modality:
        assert char in mapping.keys()
        input_size.append(mapping[char])

    encoder = get_encoder(opt, input_size)
    joint_representation_learner = get_joint_representation_learner(opt)
    have_auxiliary_tasks = sum([(1 if item not in ['lang'] else 0) for item in opt['crit']])
    auxiliary_task_predictor = get_auxiliary_task_predictor(opt)
    decoder = get_decoder(opt)
    tgt_word_prj = nn.Linear(opt["dim_hidden"], opt["vocab_size"], bias=False)

    model = Seq2Seq(
        opt=opt,
        encoder=encoder,
        joint_representation_learner=joint_representation_learner,
        auxiliary_task_predictor=auxiliary_task_predictor,
        decoder=decoder,
        tgt_word_prj=tgt_word_prj,
        )
    return model


def load_model_and_opt(checkpoint_path, device, return_other_info=False):
    checkpoint = torch.load(checkpoint_path)
    opt = checkpoint['settings']
    model = get_model(opt)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    if not return_other_info:
        return model, opt
    checkpoint.pop('state_dict')
    return model, opt, checkpoint


def get_forword_results(opt, model, data, device, only_data=False, vocab=None, **kwargs):
    category, labels = map(
        lambda x: x.to(device),
        [data['category'], data['labels']]
    )

    feats = []
    for char in opt['modality'].lower():
        feat = data.get("feats_%s" % char, None)
        assert feat is not None
        feats.append(feat.to(device))

    if opt['visual_word_generation']:
        tokens = [data['tokens_1'].to(device), data['tokens'].to(device)]
    else:
        tokens = data['tokens'].to(device)

    if only_data:
        # for evaluation
        results = model.encode(feats=feats)
    else:
        results = model(
            feats=feats,
            tgt_tokens=tokens,
            category=category,
            opt=opt,
            vocab=vocab,
            **kwargs
        )

    if opt['decoding_type'] == 'NARFormer':
        if data.get('length_target', None) is None:
            data['length_target'] = None
        data['length_target'] = data['length_target'].to(device)
        results[Constants.mapping['length'][1]] = data['length_target']
        start_index = 0
    else:
        start_index = 1

    if opt['visual_word_generation']:
        results[Constants.mapping['lang'][1]] = [
            data['labels_1'].to(device)[:, start_index:],
            labels[:, start_index:]
        ]
    else:
        results[Constants.mapping['lang'][1]] = labels[:, start_index:]

    if only_data:
        return results, category, labels
    return results