
# 自分自身のパスを取得
SCRIPT_DIR=$(cd $(dirname $0); pwd)
echo "スクリプトのディレクトリ: $SCRIPT_DIR"

# 一つ上のディレクトリに移動
cd $SCRIPT_DIR/..
TEST_DIR=$(pwd)
echo "テストディレクトリ: $TEST_DIR"

# python3 -m unittest でテストを実行する
python3 -m unittest tests/test_*.py

echo "テストが完了しました"

exit 0
