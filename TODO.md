C:\Users\gd3470\micromamba\envs\train\Lib\site-packages\numpy\_core\_methods.py:144: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)
Traceback (most recent call last):
  File "C:\Users\gd3470\Desktop\ssl\verifymatch\train.py", line 1077, in <module>
    eval_loss, eval_acc = evaluate(dev_dataset)
                          ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\gd3470\Desktop\ssl\verifymatch\train.py", line 1032, in evaluate
    return eval_loss / len(eval_loader), eval_acc