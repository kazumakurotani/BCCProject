
class Flag:
    """
    flagを管理するクラス
    """
    def __init__(self):
        """
        flagの初期化
        """
        # 動作環境の指定
        self.is_laptop_or_desktop = 1 # 1:laptop, 2:desktop

        # 
        self.is_first_call = True

