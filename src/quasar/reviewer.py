"""Prepare official data, run the main-table models and write per-question scores and tables from one path configuration."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from quasar.reviewer_data import load_official_questions, read_complete_predictions, verify_backbone

VERSION = 'quasar-release-candidate-v43-20261009-r1'
ROLES = ('base', 'visual_sft', 'visual_grpo', 'quasar_sft', 'quasar_grpo')


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = ''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in rows)
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f'Frozen run input changed: {path}; use a new run_root')
    else:
        partial = path.with_suffix(path.suffix+'.partial')
        partial.write_text(content)
        partial.replace(path)


def execute(output: Path, name: str, argv: list[str]):
    """每次执行保留独立日志；退出码非零直接暴露，不写成功标记。"""
    output.mkdir(parents=True, exist_ok=True)
    attempt = 1 + len(list(output.glob(name+'.*.command.json')))
    prefix = output / f'{name}.{attempt:03d}'
    Path(str(prefix)+'.command.json').write_text(json.dumps(argv, indent=2)+'\n')
    with Path(str(prefix)+'.log').open('w') as log:
        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
    Path(str(prefix)+'.exit_code').write_text(str(result.returncode)+'\n')
    if result.returncode:
        raise RuntimeError(f'{name} failed ({result.returncode}); see {prefix}.log')


def prepare(source, paths, dataset, selected, frame_specs, output, *, spatial, public_vsi=False, rgb_features=True):
    """原视频先复原历史取帧序列，再分别选择几何输入序列与模型输入序列。"""
    work = output / ('spatial' if spatial else 'rgb')
    work.mkdir(parents=True, exist_ok=True)
    needed = list(dict.fromkeys(x['input']['scene']['video_id'] for x in selected))
    extracted = []
    for video in needed:
        spec = frame_specs[video]
        media = Path(paths['vsti']['video_root'])/spec['media_relative_path'] if dataset == 'vsti' else (
            Path(paths['vsi']['video_roots'][video.split('__')[0]])/Path(spec['media_relative_path']).name)
        if not media.is_file():
            raise FileNotFoundError(f'Missing original video: {media}')
        folder = output/'frames'/video
        plan = dict(video_id=video, video_path=str(media), expected_frame_ids=spec['export_frame_ids'],
                    image_long_edge=spec['image_long_edge'], selection='recorded_model_input',
                    expected_total_frames=spec['expected_total_frames'])
        plan_path = output/'plans'/(video+'.jsonl')
        write_jsonl(plan_path, [plan])
        manifest = folder/'video_frame_manifest.jsonl'
        if not manifest.exists():
            if folder.exists():
                # 提取中断的目录留存，下一次使用新的尝试目录；不删除失败证据。
                number = 1
                while folder.with_name(video+f'.incomplete_{number:03d}').exists():
                    number += 1
                folder.rename(folder.with_name(video+f'.incomplete_{number:03d}'))
            execute(work/'logs', 'frames_'+video, [sys.executable, '-I', '-B', '-m', 'quasar.extract_public_frames',
                '--video-plan', str(plan_path), '--output-root', str(folder), '--num-frames', str(len(spec['export_frame_ids']))])
        records = [json.loads(x) for x in manifest.read_text().splitlines()]
        if [x['frame_id'] for x in records] != spec['export_frame_ids'] or any(not Path(x['frame_image_path']).is_file() for x in records):
            raise ValueError(f'Incomplete or changed extracted frames: {video}')
        extracted.extend(records)
    images = {(x['video_id'], x['frame_id']): x['frame_image_path'] for x in extracted}
    if spatial:
        frame_rows = []
        plans = []
        qa = []
        for video in needed:
            for i, frame_id in enumerate(frame_specs[video]['geometry_frame_ids']):
                frame_rows.append(dict(video_id=video, frame_candidate_id=f'{video}__frame_{frame_id:06d}',
                    frame_id=frame_id, sample_order=i, embedding_row_index=i, frame_image_path=images[video,frame_id]))
        write_jsonl(work/'geometry_frames.jsonl', frame_rows)
        geometry = work/'geometry'
        from quasar.export_observation_geometry import validate_existing_npz, sanitize_filename
        pending = []
        for video in needed:
            path = geometry/'video_npz'/(sanitize_filename(video)+'.npz')
            if path.exists():
                validate_existing_npz(npz_path=path,video_id=video,
                    frame_rows=[x for x in frame_rows if x['video_id']==video])
            else:pending.append(video)
        if pending:
            execute(work/'logs', 'geometry', [sys.executable, '-I', '-B', '-m', 'quasar.export_observation_geometry',
                '--video-frame-manifest', str(work/'geometry_frames.jsonl'), '--output-root', str(geometry),
                '--cut3r-repo-root', paths['cut3r']['repo'], '--cut3r-model-path', paths['cut3r']['weight'],
                '--image-size', '512', '--device', 'cuda:0', '--skip-existing'])
        from quasar.train_native_qwen3vl2b_glow_patch_spatial import parse_target_frame_index
        for record in selected:
            row = record['label']
            video = row['scene']['video_id']
            qa.append(dict(source_row=record['source_row'], sample_id=row['sample_id'], source_tag=row['source']['tag'],
                           source_dataset=row['source']['dataset'], video_id=video, question=row['question'], answer=row['answer']))
            plans.append(dict(source_row=record['source_row'], sample_id=row['sample_id'], video_id=video,
                              scene_id=row['scene']['scene_id'], geometry_source='cut3r',
                              model_frame_ids=frame_specs[video]['model_frame_ids'],
                              target_frame_index=parse_target_frame_index(row['question']['text'], max_frames=32)))
        write_jsonl(work/'qa.jsonl', qa)
        write_jsonl(work/'plan.jsonl', plans)
        reference_args = []
        if public_vsi:
            # 定位进程只接收题文与身份；答案和选项均不在其输入文件中。
            station_inputs = [dict(source_row=x['source_row'], sample_id=x['input']['sample_id'],
                video_id=x['input']['scene']['video_id'], question_text=x['input']['question']['text'],
                question_type=x['input']['question']['canonical_question_type']) for x in selected]
            write_jsonl(work/'station_inputs.jsonl', station_inputs)
            execute(work/'logs', 'locate', [sys.executable, '-I', '-B', '-m', 'quasar.vsi_reference',
                '--input-jsonl', str(work/'station_inputs.jsonl'), '--geometry-root', str(geometry),
                '--qwen-model-path', paths['backbone'], '--cut3r-repo', paths['cut3r']['repo'],
                '--output-root', str(work/'reference_estimates')])
            state = json.loads((work/'reference_estimates/STATUS.json').read_text())
            if state['status'] == 'BLOCKED_INPUT':
                from quasar.vsi_reference import ReferenceInputBlocked
                raise ReferenceInputBlocked('Unresolved station estimates; see '+str(work/'reference_estimates/STATUS.json'))
            reference_args = ['--reference-jsonl', str(work/'reference_estimates/references.jsonl')]
        inference = work/'prepared/inference.jsonl'
        if not inference.exists():
            # 参照生成可能在落盘时中断；保留这次未完成的目录，再完整生成一次。
            if inference.parent.exists():
                number = 1
                while work.joinpath(f'prepared.incomplete_{number:03d}').exists():
                    number += 1
                inference.parent.rename(work/f'prepared.incomplete_{number:03d}')
            execute(work/'logs', 'reference', [sys.executable, '-I', '-B', '-m', 'quasar.prepare_public_inputs',
                '--qa-jsonl', str(work/'qa.jsonl'), '--input-plan', str(work/'plan.jsonl'),
                '--frame-manifest', str(work/'geometry_frames.jsonl'), '--geometry-root', str(geometry),
                '--output-root', str(work/'prepared'), '--max-visual-frames', '32', *reference_args])
        cache = work/'features'
        # 不同模型共用本次生成的特征。缺件才启动导出器，已有文件仍由下方实际读取器检查。
        from quasar.train_native_qwen3vl2b_glow_patch_spatial import cache_identity_from_row, visual_cache_path_for_key
        cached_rows = [json.loads(x) for x in inference.read_text().splitlines()]
        keys = [cache_identity_from_row(row, scannet_frame_root=None, max_visual_frames=32,
                    qwen_model_path=Path(paths['backbone']), max_pixels=50176)[0] for row in cached_rows]
        if any(not visual_cache_path_for_key(cache,key).is_file() for key in keys):
            execute(work/'logs', 'features', [sys.executable, '-I', '-B', '-m', 'quasar.export_visual_cache',
                '--input-jsonl', str(inference), '--output-root', str(cache), '--qwen-model-path', paths['backbone'],
                '--max-visual-frames', '32', '--max-pixels', '50176', '--depth-grid-h', '24', '--depth-grid-w', '32',
                '--shard-index', '0', '--num-shards', '1', '--device', 'cuda:0', '--inference-only'])
    else:
        records = []
        for record in selected:
            row = record['input']
            video = row['scene']['video_id']
            row['inputs']['image_paths'] = [images[video, f] for f in frame_specs[video]['model_frame_ids']]
            records.append(row)
        inference = work/'inference.jsonl'
        write_jsonl(inference, records)
        cache = work/'features'
        if rgb_features:
            execute(work/'logs', 'features', [sys.executable, '-I', '-B', '-m', 'quasar.reviewer_rgb',
                '--input-jsonl', str(inference), '--output-root', str(cache), '--qwen-model-path', paths['backbone'], '--device', 'cuda:0'])
    # 特征导出器的单条失败记录不等于成功；用模型实际读取方式检查每个条目。
    from quasar.train_native_qwen3vl2b_glow_patch_spatial import audit_visual_cache_rows
    inference_rows = [json.loads(x) for x in inference.read_text().splitlines()]
    if spatial or rgb_features:
        audit_visual_cache_rows(rows=inference_rows, visual_cache_root=cache, scannet_frame_root=None,
            max_visual_frames=32, qwen_model_path=Path(paths['backbone']), max_pixels=50176, split_name=dataset)
    else:
        from quasar.public_vsi_base import public_messages
        for row in inference_rows:
            public_messages(row)
    if [x['sample_id'] for x in inference_rows] != [x['input']['sample_id'] for x in selected]:
        raise ValueError('Prepared inference membership changed')
    if any(set(x['answer']) - {'type','options','numeric_unit'} for x in inference_rows):
        raise ValueError('Gold answer in prepared inference input')
    write_jsonl(work/'labels.jsonl', [x['label'] for x in selected])
    return inference, cache


def generate_base(selected, inference, cache, output, paths):
    """沿Base历史提示、精度和生成函数，先做新RGB与新特征的一致性检查。"""
    from quasar.base_vsti.prompt import build_direct_native_prompt
    from quasar.train_native_qwen3vl2b_glow_patch_spatial import cache_identity_from_row, visual_cache_path_for_key
    rows = [json.loads(x) for x in inference.read_text().splitlines()]
    public = []
    for row in rows:
        key, _ = cache_identity_from_row(row, scannet_frame_root=None, max_visual_frames=32,
                                        qwen_model_path=Path(paths['backbone']), max_pixels=50176)
        feature = visual_cache_path_for_key(cache,key).with_name(key+'.rgb.pt')
        public.append(dict(sample_id=row['sample_id'], prompt=build_direct_native_prompt(
            question_payload=row['question'],answer=row['answer'],answer_type=row['answer']['type']),
            image_paths=row['inputs']['image_paths'], cache_key=key, feature_path=str(feature)))
    base = output/'base_run'
    write_jsonl(base/'inputs/public.jsonl', public)
    protocol = dict(model_path=paths['backbone'],sample_order=[x['sample_id'] for x in public],
                    matmul_allow_tf32=False,cudnn_allow_tf32=True,max_pixels=50176,attention='sdpa')
    (base/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (base/'ASSET_AUDIT.json').write_text(json.dumps({'acceptance_ids':protocol['sample_order'][:2]})+'\n')
    if not (base/'ACCEPTANCE.json').exists():
        execute(output/'logs','base_acceptance',[sys.executable,'-I','-B','-m','quasar.base_vsti.inference',
            '--run-dir',str(base),'--attempt',str(base/'acceptance'),'--acceptance'])
        (base/'ACCEPTANCE.json').write_bytes((base/'acceptance/ACCEPTANCE.json').read_bytes())
    execute(output/'logs','base_generate',[sys.executable,'-I','-B','-m','quasar.base_vsti.inference',
        '--run-dir',str(base),'--attempt',str(base/'generation'),'--shard','0','--shards','1'])
    records=read_complete_predictions(base/'predictions/shard_0.jsonl',protocol['sample_order'],require_complete=True)
    write_jsonl(output/'predictions.jsonl', records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths', type=Path, required=True)
    p.add_argument('--dataset', choices=['all','vsi','vsti'], default='all')
    p.add_argument('--role', choices=['all',*ROLES], default='all')
    p.add_argument('--stage', choices=['check','prepare','evaluate','table'], default='evaluate')
    p.add_argument('--smoke', action='store_true', help='Fixed input-route examples, not a paper score.')
    p.add_argument('--protocol', choices=['historical', 'public-vsi-v1'], default='historical')
    a = p.parse_args()
    paths = json.loads(a.paths.read_text())
    anchor = a.paths.resolve().parent
    # 路径只相对这一份配置解析，不依赖启动时的工作目录。
    for key in ['source_root','run_root','backbone']:
        paths[key] = str((anchor/paths[key]).resolve())
    for ds in ['vsi','vsti']:
        paths[ds]['qa'] = str((anchor/paths[ds]['qa']).resolve())
    paths['vsti']['video_root'] = str((anchor/paths['vsti']['video_root']).resolve())
    paths['vsi']['video_roots'] = {k:str((anchor/v).resolve()) for k,v in paths['vsi']['video_roots'].items()}
    paths['weights'] = {k:str((anchor/v).resolve()) for k,v in paths['weights'].items()}
    paths['cut3r'] = {k:str((anchor/v).resolve()) for k,v in paths['cut3r'].items()}
    source=Path(paths['source_root'])
    contracts=json.loads((source/'configs/table12_sources.json').read_text())
    profiles=json.loads((source/'configs/table12_profiles.json').read_text())
    public_config = None
    if a.protocol == 'public-vsi-v1':
        public_config = json.loads((source/'configs/public_vsi_v1.json').read_text())
        for key, value in public_config['cells'].items():
            profiles[key] = dict(profiles[key], **value)
    recipes={x['role']:x for x in json.loads((source/'configs/commands.json').read_text())['commands']}
    root=Path(paths['run_root'])/('smoke' if a.smoke else 'full')
    root.mkdir(parents=True,exist_ok=True)
    frozen={'version':VERSION,'paths':paths,'contracts':contracts,'profiles':profiles,
            'recipes': {cell['recipe']: recipes[cell['recipe']] for cell in profiles.values()
                        if isinstance(cell, dict) and 'recipe' in cell and not cell['recipe'].startswith('base_')}, 'smoke':a.smoke}
    if public_config is not None:
        frozen.update(protocol=a.protocol, public_config=public_config)
    contract_path=root/'RUN_CONTRACT.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != frozen:
        raise ValueError('Run contract changed; select a new run_root')
    contract_path.write_text(json.dumps(frozen,indent=2)+'\n')
    results=[]
    for dataset in (['vsti','vsi'] if a.dataset=='all' else [a.dataset]):
        contract=contracts[dataset]
        all_rows=load_official_questions(Path(paths[dataset]['qa']),dataset,contract)
        by_id={x['input']['sample_id']:x for x in all_rows}
        members=list(csv.DictReader((source/contract['members']).open(),delimiter='\t'))
        if len(members)!=contract['count'] or set(x['sample_id'] for x in members)!=set(by_id):
            raise ValueError('Official QA does not match the full frozen member manifest')
        ordered=[by_id[x['sample_id']] for x in members]
        for i,row in enumerate(ordered,1):
            row['source_row']=i
        frames={x['video_id']:x for x in (json.loads(l) for l in (source/contract['frames']).read_text().splitlines())}
        for role in (ROLES if a.role=='all' else [a.role]):
            cell=profiles[dataset+'/'+role]
            new_vsi = a.protocol == 'public-vsi-v1' and dataset == 'vsi' and role in {'base', 'quasar_sft', 'quasar_grpo'}
            role_ordered = ordered
            if new_vsi and role == 'base':
                base_rows = load_official_questions(Path(paths[dataset]['qa']),dataset,contract,vsi_base=True)
                base_by_id = {x['input']['sample_id']:x for x in base_rows}
                role_ordered = [base_by_id[x['sample_id']] for x in members]
                for i, row in enumerate(role_ordered, 1):
                    row['source_row'] = i
            smoke_ids = public_config['smoke_ids'] if new_vsi else contract['smoke_ids']
            selected=[x for x in role_ordered if x['input']['sample_id'] in smoke_ids] if a.smoke else role_ordered
            blockers=cell['blockers'] if not a.smoke else cell['smoke_blockers']
            info={'cell':dataset+'/'+role,'paper_table':1 if dataset=='vsti' else 2,
                  'paper_score_percent':cell['paper_score_percent'],'smoke':a.smoke,'expected_count':len(selected)}
            if new_vsi:
                info.update(input_protocol=a.protocol, historical_input_equivalence=False)
            if blockers:
                results.append(dict(info,status='BLOCKED_INPUT',blockers=blockers))
                continue
            if role!='base':
                weight=Path(paths['weights'][role])
                spec=json.loads((weight/'model_spec.json').read_text())
                if spec['id']!=cell['weight_id'] or spec['step']!=cell['step'] or spec['base_revision']!=profiles['base_revision']:
                    raise ValueError(f'Wrong checkpoint role/version: {weight}')
                if not (weight/spec['checkpoint']).is_file():
                    raise FileNotFoundError(weight/spec['checkpoint'])
            verify_backbone(Path(paths['backbone']), profiles['base_revision'])
            if role.startswith('quasar'):
                for file in [Path(paths['cut3r']['repo'])/'src/dust3r/model.py', Path(paths['cut3r']['weight'])]:
                    if not file.is_file():raise FileNotFoundError(file)
                import importlib.util
                if importlib.util.find_spec('curope') is None:
                    raise ModuleNotFoundError('Install the CUT3R curope component using docs/ENVIRONMENT.md')
            for video in dict.fromkeys(x['input']['scene']['video_id'] for x in selected):
                s=frames[video]
                media=Path(paths['vsti']['video_root'])/s['media_relative_path'] if dataset=='vsti' else Path(paths['vsi']['video_roots'][video.split('__')[0]])/Path(s['media_relative_path']).name
                if not media.is_file():
                    raise FileNotFoundError(media)
            if a.stage=='check':
                results.append(dict(info,status='INPUTS_PRESENT',full_matrix_verified=False))
                continue
            output=root/dataset/role
            output.mkdir(parents=True,exist_ok=True)
            ids=[x['input']['sample_id'] for x in selected]
            if a.stage!='table':
                existing=read_complete_predictions(output/'predictions.jsonl',ids,require_complete=False)
                if len(existing)!=len(ids):
                    from quasar.vsi_reference import ReferenceInputBlocked
                    try:
                        inference,cache=prepare(source,paths,dataset,selected,frames,
                            root/dataset/('prepared_public_base' if new_vsi and role=='base' else
                                          'prepared_public_quasar' if new_vsi else 'prepared'),
                            spatial=role.startswith('quasar'), public_vsi=new_vsi and role.startswith('quasar'),
                            rgb_features=not (new_vsi and role=='base'))
                    except ReferenceInputBlocked as error:
                        results.append(dict(info,status='BLOCKED_INPUT',blockers=[str(error)]))
                        continue
                    if a.stage=='prepare':
                        results.append(dict(info,status='PREPARED'))
                        continue
                    if new_vsi and role=='base':
                        execute(output/'logs','generate',[sys.executable,'-I','-B','-m','quasar.public_vsi_base',
                            '--input-jsonl',str(inference),'--qwen-model-path',paths['backbone'],'--output-root',str(output)])
                    elif role=='base':
                        generate_base(selected,inference,cache,output,paths)
                    else:
                        recipe=recipes[cell['recipe']]
                        argv=[]
                        for token in recipe['argv']:
                            if token.startswith('${'):
                                name=token[2:-1]
                                if name.startswith('EVAL_JSONL'):value=inference
                                elif name.startswith('OUTPUT_DIR'):value=output
                                elif name.startswith('VISUAL_CACHE_ROOT'):value=cache
                                elif name=='QWEN_MODEL_PATH':value=paths['backbone']
                                elif name=='SCANNET_FRAME_ROOT':
                                    value=root/'empty_depth';value.mkdir(exist_ok=True)
                                elif name in {'VISUAL_SFT_46592','QUASAR_SFT_44544',
                                              'DIRECTVISUAL_GRPO_FINAL_2048','DECODERAUX_GRPO_ALLTASK_FINAL_2048'}:
                                    value=paths['weights'][role]
                                else:raise ValueError(f'Unknown recipe path parameter: {name}')
                                argv.append(str(value))
                            else:argv.append(token)
                        execute(output/'logs','generate',[sys.executable,'-I','-B','-m',recipe['module'],*argv,
                                                          '--inference-only','--resume-inference'])
            predictions=read_complete_predictions(output/'predictions.jsonl',ids,require_complete=True)
            from quasar.retained_predictions_score import rescore_row,aggregate
            scored=[rescore_row(pred,item['label'],dataset.upper()) for pred,item in zip(predictions,selected)]
            write_jsonl(output/'scores.jsonl',scored)
            score=aggregate(scored, 'vsti_5task' if dataset=='vsti' else 'vsi_8task')
            result=dict(info,status='SMOKE_PASSED' if a.smoke else 'EVALUATED',score=score,
                        paper_delta_pp=None if a.smoke else score['score_percent']-cell['paper_score_percent'],
                        predictions=str(output/'predictions.jsonl'),source='new_model_generation',
                        paper_source=cell['paper_summary'])
            (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
            results.append(result)
    # 少量样例永远不填入主表成绩；完整表要求每格均有完整逐题回答。
    (root/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
    lines=['| Table | Model | Status | N | Score (%) | Paper (%) | Difference (pp) |',
           '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for x in results:
        score=x.get('score',{}).get('score_percent') if not a.smoke else None
        delta=x.get('paper_delta_pp')
        lines.append(f'| {x["paper_table"]} | {x["cell"]} | {x["status"]} | {x["expected_count"]} | '
                     f'{"—" if score is None else format(score,".6f")} | {x["paper_score_percent"]} | '
                     f'{"—" if delta is None else format(delta,"+.6f")} |')
    (root/'TABLE12.md').write_text('\n'.join(lines)+'\n')
    # 两张表按论文列顺序输出；样例只有运行状态，不生成可误认为主表的成绩。
    columns={1: ['camera_object_absolute_distance','camera_displacement','camera_movement_direction',
                 'object_object_relative_position','camera_object_relative_distance'],
             2: ['object_counting','object_abs_distance','object_size_estimation','room_size_estimation',
                 'object_rel_distance','object_rel_direction','route_planning','obj_appearance_order']}
    for table, tasks in columns.items():
        with (root/f'TABLE{table}.tsv').open('w') as handle:
            names=['model','status','count','average','paper_average','difference_pp',*tasks,
                   *[task+'_difference_pp' for task in tasks],'source','paper_source']
            writer=csv.DictWriter(handle,fieldnames=names,delimiter='\t')
            writer.writeheader()
            for result in results:
                if result['paper_table']!=table:continue
                values={'model':result['cell'],'status':result['status'],'count':result['expected_count'],
                        'paper_average':result['paper_score_percent'],
                        'source':result.get('source','unavailable'),
                        'paper_source':profiles[result['cell']]['paper_summary']}
                if result['status']=='EVALUATED':
                    details=result['score']['details']
                    scores=details['group_scores'] if table==1 else details['task_scores']
                    if set(scores)!=set(tasks):raise ValueError('Incomplete benchmark task coverage')
                    values.update(average=result['score']['score_percent'],difference_pp=result['paper_delta_pp'])
                    for task in tasks:
                        values[task]=scores[task]*100
                        values[task+'_difference_pp']=(scores[task]-profiles[result['cell']]['paper_task_scores'][task])*100
                writer.writerow(values)
    print(json.dumps(results,indent=2))
    if any(x['status']=='BLOCKED_INPUT' for x in results):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
