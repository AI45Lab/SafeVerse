SOURCE_DIR="base model path"
TARGET_DIR="target path"
LOCAL_DIR="checkpoint path"


python ./model_merger.py \
    --backend fsdp \
    --local_dir  $LOCAL_DIR \
    --target_dir $TARGET_DIR \
    --hf_model_path $SOURCE_DIR

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误：源目录不存在: $SOURCE_DIR"
    exit 1
fi

# 检查目标目录是否存在，不存在则创建
if [ ! -d "$TARGET_DIR" ]; then
    echo "目标目录不存在，正在创建: $TARGET_DIR"
    mkdir -p "$TARGET_DIR"
fi

# 定义要复制的基础配置文件列表
CONFIG_FILES=(
    "added_tokens.json"
    "config.json"
    "tokenizer_config.json"
    "tokenizer.json"
    "vocab.json"
    "special_tokens_map.json"
    "generation_config.json"
    "preprocessor_config.json"
    "training_args.bin"
    "merges.txt"          # 如果你有 BPE 分词器
    "chat_template.json"  # 如果你用的是对话模板
)

# 开始复制文件
echo "开始复制基础模型配置文件到 $TARGET_DIR ..."
for file in "${CONFIG_FILES[@]}"; do
    if [ -f "$SOURCE_DIR/$file" ]; then
        cp "$SOURCE_DIR/$file" "$TARGET_DIR/"
        echo "✅ 已复制: $file"
    else
        echo "⚠️ 未找到: $file"
    fi
done

echo "🎉 所有基础配置文件复制完成！"