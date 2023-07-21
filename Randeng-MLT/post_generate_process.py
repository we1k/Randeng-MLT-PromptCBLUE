# coding=utf-8
# Created by Michael Zhu
# ECNU, 2023

import json
import sys

# NOTE: no extra Python dependency package are allowed

str2Roman = {
    "I": "Ⅰ",
    "II": "Ⅱ",
    "III": "Ⅲ",
    "IV": "Ⅳ",
    "V": "Ⅴ",
    "VI": "Ⅵ",
    "VII": "Ⅶ",
    "VIII": "Ⅷ",
    "IX": "Ⅸ",
    "X": "Ⅹ",
    "XI": "Ⅺ",
    "XII": "Ⅻ"
}
    
Roman2str = {
    "Ⅰ": "I",
    "Ⅱ": "II",
    "Ⅲ": "III",
    "Ⅳ": "IV",
    "Ⅴ": "V",
    "Ⅵ": "VI",
    "Ⅶ": "VII",
    "Ⅷ": "VIII",
    "Ⅸ": "IX",
    "Ⅹ": "X",
    "Ⅺ": "XI",
    "Ⅻ": "XII",
}
# Sort the mapping rules by length in descending order
str2Roman = dict(sorted(str2Roman.items(), key=lambda item: len(item[0]), reverse=True))
Roman2str = dict(sorted(Roman2str.items(), key=lambda item: len(item[1]), reverse=True))

def convert_str2roman(sent):
    for str_roman, unicode_roman in str2Roman.items():
        sent = sent.replace(str_roman, unicode_roman)
    return sent

def convert_roman2str(sent):
    for unicode_roman, str_roman in Roman2str.items():
        sent = sent.replace(unicode_roman, str_roman)
    return sent

def convert_eng2chn(sent):
    return sent.replace(',', "，").replace('(','（').replace(')', '）')

def convert_chn2eng(sent):
    return sent.replace('，', ',').replace('（', '(').replace('）', ')')

def contains_roman_keys(sentence):
    for key in Roman2str.keys():
        if key in sentence:
            return True
    return False

def prefix_match(target, word_list):
    return [word for word in word_list if word.startswith(target) and len(word) >= 5]

def partial_match(target, word_list):
    return [word for word in word_list if target in word]

def process_generated_results(pred_file):

    structured_output = {
        "CMeEE-V2": [],
        "CMeIE": [],
        "CHIP-CDN": [],
        "CHIP-CDEE": [],
        "CHIP-STS": [],
        "CHIP-CTC": [],
        "CHIP-MDCFNPC": [],
        "KUAKE-IR": [],
        "KUAKE-QIC": [],
        "KUAKE-QQR": [],
        "KUAKE-QTR": [],
        "MedDG": [],
        "IMCS-V2-MRG": [],
        "IMCS-V2-NER": [],
        "IMCS-V2-DAC": [],
        "IMCS-V2-SR": [],
    }
    
    
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)

            sample_id_ = line.get("sample_id", "xxxx")
            input = line["input"]
            gen_output = line["target"]
            gen_output = gen_output.replace(":", "：", 100).replace(",", "，", 100).replace(";", "；", 100)
            if contains_roman_keys(input):
                # replace roman keys:
                for str_roman, unicode_roman in str2Roman.items():
                    gen_output = gen_output.replace(str_roman, unicode_roman)
                
            # gen_output = line["generated_output"]
            task_dataset = line["task_dataset"]
            task_type = line["task_type"]

            answer_choices = line["answer_choices"]
            
            
            if task_dataset == "CMeEE-V2":

                list_entities = []
                assert isinstance(answer_choices, list)
                for choice in answer_choices:
                    for piece in gen_output.split(" "):
                        if piece.startswith(f"{choice}实体"):
                            mentions = piece.replace(f"{choice}实体：", "").split("，")
                            # reconstruct mentions
                            reconstruct_mentions = []
                            cur_ent = ""
                            flag = False
                            for ans in mentions:
                                if flag == False:
                                    if '(' in ans and ')' not in ans:
                                        flag = True
                                        cur_ent = ans
                                    else:
                                        reconstruct_mentions.append(ans)
                                elif flag == True:
                                    cur_ent += ',' + ans
                                    if ')' in ans:
                                        flag = False
                                        reconstruct_mentions.append(cur_ent)
                                        cur_ent = ''
                            
                            # if (set(reconstruct_mentions) != set(mentions)):
                            #     print(sample_id_, 'CMEEE', reconstruct_mentions)
                            
                            # convert CMeEE eng2chn
                            mentions = [convert_eng2chn(m) for m in reconstruct_mentions]
        
                            mentions = list(set(mentions))
                            
                            mentions = [w.strip() for w in mentions if len(w.strip()) > 0]
                            for ment in mentions:
                                if choice == '临床表现' and len(ment) > 40: continue
                                # if choice == '医疗程序' and len(ment) > 30: continue
                                
                                list_entities.append(
                                    {
                                        "entity": ment,
                                        "type": choice,
                                        # "sample_id": sample_id_,
                                    }
                                )

                structured_output["CMeEE-V2"].append(
                    {
                        "sample_id": sample_id_,
                        "answer": list_entities,
                    }
                )

            elif task_dataset == "CMeIE":

                list_spos = []
                assert isinstance(answer_choices, list)
                list_answer_strs = gen_output.split("。 ")
                answer_hash = {}
                
                for line in list_answer_strs:
                    # 可能存在多个关系
                    multi_pred = line.split("关系： ")
                    line = multi_pred[-1]
                    if line.endswith('关系：'): continue
                    if line.endswith('。'): line = line[:-1]
                    predicate = line.split("关系：")[0].strip()
                    line = line.replace(f"{predicate}关系：", "")
                    for spo_str in line.split("。"):
                        if len(spo_str) <= 3:
                            print(sample_id_, "wrong")
                            continue
                        
                        if len(spo_str) > 3 and spo_str[0] == '(' and spo_str[-1] == ')':
                            spo_str = spo_str[1:-1]
                            head_mention = spo_str.split('，')[0].strip()
                            if len(spo_str.split('，')) < 2: continue
                            for tail_mention in spo_str.split('，')[1][1:-1].split('|'):
                                
                                # 去重
                                if (predicate, head_mention) not in answer_hash:
                                    answer_hash[(predicate, head_mention)] = [tail_mention]
                                else:
                                    if tail_mention in answer_hash[(predicate, head_mention)]: continue
                                
                                
                                # Added bracket mapping
                                if '(' in predicate or ')' in predicate:
                                    predicate = predicate.replace('(', '（').replace(')', '）')
                                list_spos.append(
                                {
                                    "predicate": predicate,
                                    "subject": head_mention,
                                    "object": tail_mention,
                                }
                            )
                        
                        if len(spo_str.split("，尾实体为")) < 2:
                            continue

                     
                structured_output[f"{task_dataset}"].append(
                    {
                        "sample_id": sample_id_,
                        "answer": list_spos,
                    }
                )

            elif task_dataset == "CHIP-CDN":
                # 答案格式：
                #   多个选中的标准化实体，用 ， 符号分割
                # print(answer_str)
                answer_str = gen_output.split(" ")[-1]
                answers = answer_str.split("，")
                reconstruct_answers = []
                
                # reconstruct answers: may contains \"\"
                cur_ent = ""
                flag = False
                for ans in answers:
                    if flag == False :
                        if '\"' in ans:
                            flag = True
                            cur_ent = ans
                        else:
                            reconstruct_answers.append(ans)
                    elif flag == True:
                        cur_ent += "," + ans
                        if '\"' in ans:
                            flag = False
                            reconstruct_answers.append(cur_ent)
                            cur_ent = ""

                answers = reconstruct_answers
                
                
                answers = [w.strip() for w in answers if len(w.strip()) > 0]
                ret_answers = []
                for ans in answers:
                    if convert_str2roman(ans) in answer_choices:
                        ret_answers.append(convert_str2roman(ans))
                    elif convert_roman2str(ans) in answer_choices:
                        ret_answers.append(convert_roman2str(ans))
                    elif len(prefix_match(ans, answer_choices)) > 0:
                        print("ret_answer_append", prefix_match(ans, answer_choices)[0])
                        ret_answers.append(prefix_match(ans, answer_choices)[0])
                    elif ans != "没有对应的标准化实体":
                        print(sample_id_, gen_output, ans, answer_choices)
                
                answers = list(set(ret_answers))
                
                answers = [
                    {
                        "entity": w,
                        "type": "normalization",
                        # "sample_id": sample_id_,
                    }
                    for w in answers
                ]
                structured_output["CHIP-CDN"].append(
                    {
                        "sample_id": sample_id_,
                        "answer": answers,
                    }
                )

            elif task_dataset == "CHIP-CDEE":
                # 答案格式：
                #   第一行：引导词
                #   每个事件占一行，事件字段用 ； 分隔， 然后每个字段是 字段名：字段值的格式"
                #                                  字段值有多个，则用 ，符号分隔
                keys = ["主体词", "发生状态", "描述词", "解剖部位"]

                list_answer_strs = gen_output.split(" ")
                list_events = []
                for ans_str in list_answer_strs:
                    ans_str = ans_str[1:-1]
                    event_info = {}
                    ans_attrs = ans_str.split("；")
                    for a_attr,key in zip(ans_attrs, keys):
                        a_attr = a_attr.strip()
                        if key in ["描述词", "解剖部位"]:
                            a_attr_split = a_attr.split("，")
                            a_attr_split = [w.strip() for w in a_attr_split if len(w.strip()) > 0]
                            event_info[key] = a_attr_split
                        else:
                            event_info[key] = a_attr
                            

                    for key in keys:
                        if key not in event_info:
                            if key in ["描述词", "解剖部位"]:
                                event_info[key] = []
                            else:
                                event_info[key] = ""

                    # event_info["sample_id"] = sample_id_
                    list_events.append(event_info)


                structured_output[task_dataset].append(
                    {
                        "sample_id": sample_id_,
                        "answer": list_events,
                    }
                )

            elif task_dataset == "CHIP-STS":
                answer_choices = ["是的", "不是"]
                answer_str = gen_output
                if answer_str == "是" * 10:
                    answer_str = "是的"
                elif answer_str == "不" * 10:
                    answer_str = "不是"
                else:
                    print('STS wrong', sample_id_, answer_str)


                structured_output["CHIP-STS"].append(
                    {
                        "sample_id": sample_id_,
                        "answer": answer_str,
                    }
                )


            elif task_dataset == "CHIP-CTC":
                # 答案格式：直接回答分类标签
                answer_str = gen_output.strip()
                
                answer_choices.append("体征(医生检测)")
                answer_choices.append("非上述类型")
                ans_dict = {k: 0 for k in answer_choices}
                for ans in answer_str.split('；'):
                    if ans in ans_dict.keys():
                        ans_dict[ans] += 1
                
                max_cnt = -1
                for k, v in ans_dict.items():
                    if v > max_cnt:
                        max_cnt = v
                        answer_str = k
                if answer_str == '体征(医生检测)': answer_str = "体征(医生检测）"
                
                if max_cnt == 0: answer_str = "非上述类型"
                if max_cnt != 3 or answer_str not in answer_choices:
                    print(sample_id_, max_cnt, answer_str, gen_output)
                
                
                # hard coding
                if answer_str in ["锻炼", "种族", "贮存", "睡眠"]:
                    answer_str = "非上述类型"
                    
                structured_output[task_dataset].append(
                    {
                        "sample_id": sample_id_,
                        "answer": answer_str,
                    }
                )

            elif task_dataset == "KUAKE-IR":
                # 答案格式：直接回答 "相关", "不相关"
                answer_str = gen_output.strip()
                # if answer_str not in answer_choices:
                #     answer_str = "不相关"

                ans_dict = {k: 0 for k in answer_choices}
                for ans in answer_str.split('；'):
                    if ans in ans_dict.keys():
                        ans_dict[ans] += 1
                
                max_cnt = -1
                for k, v in ans_dict.items():
                    if v > max_cnt:
                        max_cnt = v
                        answer_str = k 
                
                structured_output[task_dataset].append(
                    {
                        "sample_id": sample_id_,
                        "answer": answer_str,
                    }
                )

            elif task_dataset == "KUAKE-QIC":
                # 答案格式：直接回答分类标签
                answer_str = gen_output.strip()
                # if not answer_str in answer_choices:
                #     answer_str = "非上述类型"
                answer_choices.append("非上述类型")
                ans_dict = {k: 0 for k in answer_choices}
                for ans in answer_str.split('；'):
                    if ans in ans_dict.keys():
                        ans_dict[ans] += 1
                
                max_cnt = -1
                for k, v in ans_dict.items():
                    if v > max_cnt:
                        max_cnt = v
                        answer_str = k 
                if max_cnt == 0:
                    answer_str = "非上述类型"
                
                
                if answer_str == "疾病描述":
                    answer_str = "病情诊断"
                # print(sample_id_, gen_output, answer_str)
                
                if answer_str not in answer_choices:
                    print(answer_str, answer_choices, answer_str not in answer_choices)
                
                structured_output[task_dataset].append(
                    {
                        "sample_id": sample_id_,
                        "answer": answer_str,
                    }
                )

            elif task_dataset == "KUAKE-QQR":
                # 答案格式：直接回答分类标签
                answer_str = gen_output.strip()
                # if not answer_str in answer_choices:
                #     print(sample_id_, "QQR")

                structured_output[task_dataset].append(
                    {
                        "sample_id": sample_id_,
                        "answer": answer_str,
                    }
                )

            elif task_dataset == "KUAKE-QTR":
                # 答案格式：直接回答分类标签
                answer_str = gen_output.strip()
                # if not answer_str in answer_choices:
                #     answer_str = "完全不匹配"


                structured_output[task_dataset].append(
                    {
                        "sample_id": sample_id_,
                        "answer": answer_str,
                    }
                )

            elif task_dataset == "CHIP-MDCFNPC":
                # 答案格式：
                #   第一行：引导词
                #    每一行就是 "[症状词]：[阴阳性判断结果]"
                list_answer_strs = gen_output.split(" ")

                candidate_ent = []
                list_finding_attrs = []
                
                lines = line["input"].split('\n')
                for l in lines:
                    if l.startswith('临床发现实体：'):
                        candidate_ent = l[7:].split('，')
                
                finding_hash = {ent: [] for ent in candidate_ent}
                for ans_str in list_answer_strs:
                    if not len(ans_str.split("：")) == 2:
                        continue

                    finding, conclusion = ans_str.split("：")
                    if conclusion.startswith("阳阳阳"):
                        conclusion = "已有症状疾病或者假设未来可能发生的疾病等"
                    elif conclusion.startswith("阴阴阴"):
                        conclusion = '未患有症状疾病'
                    elif conclusion.startswith("其他"):
                        conclusion = '没有回答、不知道、回答不明确或者模棱两可不好推断'
                    elif conclusion.startswith('不标注'):
                        conclusion = '无实际意义的不标注或者和病人当前的状态独立不标注'
                    
                    
                    #  去重
                    finding = finding.strip()
                    if finding in finding_hash:
                        if len(finding_hash[finding]) == 1: continue
                        # if conclusion in finding_hash[finding]: continue
                        finding_hash[finding].append(conclusion)
                        list_finding_attrs.append(
                            {
                                "entity": finding.strip(),
                                "attr": conclusion
                            }
                        )
                        
                    else:
                        for k in finding_hash:
                            if k in finding:
                                if len(finding_hash[k]) == 1: continue
                                finding_hash[k].append(conclusion)
                                list_finding_attrs.append(
                                    {
                                        "entity": k.strip(),
                                        "attr": conclusion
                                    }
                                )
                                break
                    
                for k, v in finding_hash.items():
                    if len(v) == 0:
                        if line["input"].count(k) == 1:
                            # print(sample_id_, k.strip())
                            list_finding_attrs.append(
                                {
                                    "entity": k.strip(),
                                    "attr": "没有回答、不知道、回答不明确或者模棱两可不好推断",
                                }
                            )
                            
                        # elif  line["input"].count(k) == 2:
                        #     # TODO: FIX THIS
                        #     print(sample_id_, k, line["input"].count(k))
                        #     list_finding_attrs.append(
                        #         {
                        #             "entity": k.strip(),
                        #             "attr": "无实际意义的不标注或者和病人当前的状态独立不标注",
                        #         }
                        #     )
                        elif  line["input"].count(k) >= 2:
                            # print(sample_id_, k, line["input"].count(k))
                            list_finding_attrs.append(
                                {
                                    "entity": k.strip(),
                                    "attr": "已有症状疾病或者假设未来可能发生的疾病等",
                                }
                            )
                            

                structured_output[f"{task_dataset}"].append(
                    {
                        "sample_id": sample_id_,
                        "answer": list_finding_attrs,
                    }
                )

            elif task_dataset == "IMCS-V2-NER":
                # 答案格式：
                #   第一行：引导词
                #   实体每类占一行，每行格式为 "[类型名称]实体：实体名称1，实体名称2，实体名称3\n"
                #                多个实体，用 ， 符号分割

                list_entities = []
                assert isinstance(answer_choices, list)
                ent_answers = gen_output.split(" ")[1: ]
                
                for choice in answer_choices:
                    for piece in ent_answers:
                        if piece.startswith(f"{choice}实体"):
                            mentions = piece.replace(f"{choice}实体：", "").split("，")
                            mentions = [w.strip() for w in mentions if len(w.strip()) > 0]
                            for ment in mentions:
                                list_entities.append(
                                    {
                                        "entity": ment,
                                        "type": choice,
                                    }
                                )

                structured_output["IMCS-V2-NER"].append(
                    {
                        "sample_id": sample_id_,
                        "answer": list_entities,
                    }
                )

            elif task_dataset == "IMCS-V2-DAC":
                # 答案格式：直接回答分类标签
                answer_str = gen_output.strip()
                # if not answer_str in answer_choices:
                #     answer_str = "非上述类型"

                structured_output[task_dataset].append(
                    {
                        "sample_id": sample_id_,
                        "answer": answer_str,
                    }
                )

            elif task_dataset == "IMCS-V2-SR":
                # 答案格式：
                #   第一行：引导词
                #    每一行就是 "[症状词]：[阴阳性判断结果]"
                list_answer_strs = gen_output.split(" ")

                list_finding_attrs = []
                finding_hash = {}
                for ans_str in list_answer_strs:
                    if not len(ans_str.split("：")) == 2:
                        continue

                    finding, conclusion = ans_str.split("：")
                    if conclusion.startswith("阳阳阳"):
                        conclusion = "患有该症状"
                    elif conclusion.startswith("阴阴阴"):
                        conclusion = '没有患有该症状'
                    elif conclusion.startswith("不确定"):
                        conclusion = '无法根据上下文确定病人是否患有该症状'
                    
                    #  去重
                    finding = finding.strip()
                    if finding not in finding_hash:
                        finding_hash[finding] = [conclusion]
                        list_finding_attrs.append(
                            {
                                "entity": finding.strip(),
                                "attr": conclusion
                            }
                        )
                    
                    else:
                        if conclusion in finding_hash[finding]: continue
                            list_finding_attrs.append(
                                {
                                    "entity": finding.strip(),
                                    "attr": conclusion
                                }
                            )
                    # elif partial_match(finding, finding_hash):
                    #     pass
                        # print(sample_id_, partial_match(finding, finding_hash), finding)
       

                structured_output[f"{task_dataset}"].append(
                    {
                        "sample_id": sample_id_,
                        "answer": list_finding_attrs,
                    }
                )

            elif task_dataset == "IMCS-V2-MRG":
                # 答案格式：
                #   1. 第一行是引导词；
                #   第二行开始是 [section_name]：str的格式

                keys = [
                    "主诉：",
                    "现病史：",
                    "辅助检查：",
                    "既往史：",
                    "诊断：",
                    "建议："
                ]
                answer_dict = {}
                for key in keys:
                    for line in gen_output.strip().split(" ")[1: ]:
                        if not line.startswith(key):
                            continue
                        answer_str = line.strip().split(key)[-1].strip()
                        answer_dict[key[: -1]] = answer_str

                structured_output[f"{task_dataset}"].append(
                    {
                        "sample_id": sample_id_,
                        "answer": answer_dict,
                    }
                )

            elif task_dataset == "MedDG":
                # 答案格式：str
                gen_output = gen_output[:100].replace('?', '？').replace('你好，很高兴为您解答。', '').replace('你好，很高兴为你解答。', '')
                answer_str = gen_output.strip()

                structured_output[f"{task_dataset}"].append(
                    {
                        "sample_id": sample_id_,
                        "answer": answer_str,
                    }
                )

            else:

                raise ValueError


    return structured_output


if __name__ == "__main__":
    
    from_dir = sys.argv[1]
    to_dir = sys.argv[2]
    structured_outputs = process_generated_results(
        from_dir
    )
    for key in structured_outputs.keys():
        print(key, len(structured_outputs[key]))
    json.dump(
        structured_outputs,
        open(to_dir, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2
    )