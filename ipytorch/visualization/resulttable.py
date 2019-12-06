import os

__all__ = ["ResultTable"]

class ResultTable(object):
    """
    create package of experimental results
    """
    def __init__(self, save_path):

        self.save_path = save_path
        self.save_dir = os.path.split(save_path)[0]

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        if os.path.isfile(self.save_path):
            os.remove(self.save_path)

        print("|===>Result table will be saved at", self.save_path)
    
    def write_table_title(self, input_data):
        """
        save results table title to results.md
        """
        txt_file = open(self.save_path, 'a+')
        txt_file.write("## " + str(input_data) + "\n")
        txt_file.close()

    def write_table_head(self, *table_head):
        """
        save results head to results.md
        """
        txt_file = open(self.save_path, 'a+')
        for head in table_head:
            txt_file.write("| " + str(head) + " ")
        txt_file.write("|\n")
        for head in table_head:
            txt_file.write("| ---" + " ")
        txt_file.write("|\n")
        txt_file.close()
    
    def write_table_body(self, *table_body):
        """
        save results body to results.md
        """
        txt_file = open(self.save_path, 'a+')
        for body in table_body:
            txt_file.write("| " + str(body) + " ")
        txt_file.write("|\n")
        txt_file.close()