import string
class encoding_error(Exception):
    pass
class label_encoder:
    def fit(self,data:list,dtype=bool):
        unique = list(dict.fromkeys(data))
        encoded= []
        if len(unique)>2:
            raise encoding_error("only two unique values allowed")
        unique_d = list(unique)
        for i in data:
            if i == unique_d[0]:
                encoded.append(dtype(True))
            elif i == unique_d[1]:
                encoded.append(dtype(False))
        self.encoded = encoded
    def get_encode(self):
        return (self.encoded)
class one_hot_encoder:
    def fit(self,data:list,dtype=bool):
        unique = list(dict.fromkeys(data))
        encoded = {}
        for i in unique:
            encoded.update({i:[]})
        for i in data:
            for j in encoded:
                encoded[j].append(dtype(i==j))
        self.encoded = encoded
    def get_encoded(self):
        return self.encoded
class binary_encoder:
    def alphabets_to_bin(self,n:str):
        con = list(string.ascii_letters)
        converter = {}
        mid = []
        for i,j in zip(con,range(1,53)):
            converter.update({i:str(bin(j).replace("0b",""))})
        for i in range(0,10):
            converter.update({str(i):str(bin(i+53)).replace("0b","")})
        for z in n:
            mid.append(str(converter.get(z)))
        converter[' '] = str(bin(63)).replace("0b","")
        result = "".join(mid)
        return result
    def fit(self,data:list,dtype=bool):
        unique = list(dict.fromkeys(data))
        encoded = {}
        compare = {}
        for i in unique:
            encoded.update({self.alphabets_to_bin(i):[]})
            compare.update({i:self.alphabets_to_bin(i)})
        for i in data:
            for j in encoded:
                real_encode = compare.get(i)
                encoded[j].append(dtype(real_encode==j))
        self.encoded = encoded
    def get_encoded(self):
        return self.encoded
