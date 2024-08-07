import gc # garbege collect
from threading import Timer # threding timer
global timer

def record_once():
    # _clear_mem_cache()
    gc.collect()
    # _print_mem_free()
    
class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


if __name__ == '__main__':
 
		# global timer
		timer = RepeatTimer(15, record_once) # run record_once per 15 s
		timer.start() # run timer
		
		## run your method
		
		timer.cancel() # end timer
