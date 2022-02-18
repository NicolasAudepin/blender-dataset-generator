from threading import Thread
import time

from queue import Queue, Empty

import random
from rich.live import Live
from rich.table import Table



class Training_display_tread(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.training = True
        self.queue = Queue()
        self.data = dict()
        self.data["Empty"]=None
        self.previous_data = []
        
        
    def stop(self):
        self.training = False
        
    def setI(self,i):
        self.i = i
        print(f"{i} I")
    


    def generate_table(self):
        table = Table()
        ligne = []
        for key,value in self.data.items():
            table.add_column(key)
            ligne.append(value)

        for data in self.previous_data[-10:] :
            table.add_row(*data.values())          

        table.add_row(*ligne)
    
        return table
    
    def run(self):
        with Live(self.generate_table(), refresh_per_second=10) as live:
            
            while self.training:
                try:
                    self.data = self.queue.get(timeout=1)
                except Empty:
                    pass
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except Exception:
                    raise Exception
                live.update(self.generate_table())

                


if __name__ == "__main__":

    thread = Training_display_tread()
    thread.start()
 
    for i in range(20):
        try:
            d = dict()
            d["Epoch"]=f"[yellow]{random.randint(1,50)}"
            d["I"]=f"[#ffaa55 bold]{i}"
            
            time.sleep(0.5)
        
            thread.queue.put(d)

        except KeyboardInterrupt:
            thread.stop()
            raise KeyboardInterrupt
        except Exception:
            thread.stop()
            raise Exception
        finally:
            pass
        # print(i)

    thread.stop()
    
