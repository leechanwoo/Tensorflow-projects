
import collections
import tensorflow as tf


def main(_):
    raw_data = ['후아....... 이제서야 결과 같은 결과가 나타났다.도무지 텐서보드에 뿌려진 데이터들이 왜 그렇게 나오는지 이해가 안 됐었는데... 애초에 tensorboard에 그릴 때 serialize를 잘 못하고 있었다. 배치 unpack과 동시에 0, 1 axis에서 transpose가 일어나는 것을 생각 못하고 그대로 pack 해버리니 당연히 sin 곡선이 높은 주파수로 나왔던 것.라이브에서도 말했 듯, state도 잘 못 들어갔다. state랑 recurrent 잘 못 들어간거 다 찾아서 맞게 다 고쳤다. 이젠 모델에 대한 이슈는 없다.문제는 이제 임의의 텐서가 들어올 때 line chart를 그려주는 기능을 구현하려 했는데 기본적으로 shape이 어떻게 들어올 지 모르니 다차원 텐서랑 1차원 벡터랑 shape으로 구분해서 serialize 기능을 조건문으로 분기하려 했지만, tf.cond() 이거 너무 구리다. 본디 조건문이라 함은 if문이나 else문 중 하나만 돌아야 정상인데 아무래도 graph operation 방식이다 보니 일단 graph 그릴 땐 if, else문 다 돌아 본 다음 세션 run 할 때 비로소 if문 처럼 분기가 되도록 되어 있었다. 텐서플로는 graph 그릴 때 사이즈 연산을 꼭 해준다는 것.... 쉣또빠킹 shape이 다르다는 조건으로 분기문을 만들어놨더니 shape 틀리다고 계속 오류를 띄웠더랬다.... 어쨌든 graph를 생성해야 되는데 shape 값을 세션 들어가기 전엔 절대 모르니 일단 분기문을 다 돌아보고 shape 안 맞는 연산일 경우 에러를 띄워버리는 것...... 이거 땜에 보수시간 대부분을 삽질로 넘어가고 결국 분기문은 이렇게 사용 할 수 없다는걸로 결론 어차피 label이랑 prediction이랑 비교하기 위한 plot 기능을 구현하는거니까 아에 이 용도에 맞게 최적화를 해버렸다. 어차피 batch로 만들어진 label은 prediction이랑 shape이 같을테니까... NLP 공부해야 되는데 tensorflow로 일 다 보겠다 흑... ㅠ 하긴.. tensorboard를 이렇게 쓰라고 만든게 아니라는건 알지만서도.... 이럴거면 plot 기능을 좀 넣던가 아오 ㅠㅠㅠㅠ 그래도 뭐 ... 만족스러운 결과가 나와서 다행']

    words = collections.Counter(raw_data[0].split())
    counts = sorted(words.items(), key=lambda x: x[1], reverse=True)
    print(counts)

if __name__ == "__main__":
    tf.app.run()
