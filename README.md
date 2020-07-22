# mix-density-network-MDN
solution of the one-to-many question based on the mix density network

## renference 
[1] https://zhuanlan.zhihu.com/p/37992239
[2] https://zh.wikipedia.org/wiki/%E5%A4%9A%E5%80%BC%E5%87%BD%E6%95%B0
[3] https://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/

## one-to-many question
As is known to all, the network does well in mapping relation fundation. However, the network may struggle in the one-to-many question, which means a single x may get several different y.

## main scripts
 mix_state_network.py
 Including the NetWork class which compossed of netmodel, training and testing 3 parts.

## figures
(1) Training process
 <p align="center">
  <img src="https://github.com/guanshaoheng/mix-density-network-MDN/blob/master/traing_process.svg"/>
 </p>
(2) Distribution of the possibility, loction and bias of the normal density distribution.
<p align="center">
  <img src="https://github.com/guanshaoheng/mix-density-network-MDN/blob/master/distribution of p mu sigma.svg"/>
 </p>
(3) Mixing process
<p align="center">
  <img src="https://github.com/guanshaoheng/mix-density-network-MDN/blob/master/gaussian function mixing process.svg"/>
 </p>
(4) Prediction
<p align="center">
  <img src="https://github.com/guanshaoheng/mix-density-network-MDN/blob/master/prediction.svg"/>
 </p>
