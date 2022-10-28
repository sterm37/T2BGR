from tqdm import tqdm
import numpy as np
import gen4
import imageio
from PIL import Image, ImageChops
from torchvision.transforms import functional as TF
import torch
import os
import socket
import threading
import argparse

host = "127.0.0.1"
port = 22334
user_list = {}
notice_flag = 0
loopVal = True
lock = threading.Lock()

def handle_receive(client_socket, addr, user):

    #메시지를 받는동안 반복
    while True:
        data = bytearray(client_socket.recv(1024))[2:]
        promp = data.decode('utf-8')
        #종료 명령어
        if "/exit" in promp:
            msg = "---- %s Thread is DONE. ----" % user
            print(msg)
            del user_list[user]
            break
        print('gen')
        print(promp)
        #쓰레드 잠금
        lock.acquire()
        try:
            genImage(promp, user)
        finally:
            lock.release()
        
        msg = "%s : %s"%(user, promp)
        print("\n", str(msg), "\n")
        imgPATH = './Saves/' + str(user) + "THR.png"
        print("sending img...")
        sendIMG(client_socket, imgPATH)
        print("success send img..")

    client_socket.close()

def sendIMG(sock, imgPath):
    imgF = open(imgPath, "rb")
    data = imgF.read()
    dataLen = len(data)
    sock.sendall(dataLen.to_bytes(4, byteorder="big"))
    print("dataLEN: ", dataLen)

    step = 1024
    #loop = int(dataLen/step) + 1
    print("loop start")
    while len(data) > 0:
        print(len(data))
        if len(data) < step:
            sock.sendall(data)
            data = []
        else:
            sock.sendall(data[:step])
            data = data[step:]
    print("loop end")
    imgF.flush()
    imgF.close()

def thrGenImage():
    server_socket = socket.socket(socket.AF_INET)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(3)
    thrNum = 1

    while True:
        if loopVal == True:
            client_socket, addr = server_socket.accept()
        else:
            for user, con in user_list:
                con.close()
            server_socket.close()
            print("Keyboard interrupt")
            break
        user = str(thrNum)
        user_list[user] = client_socket
        thrNum += 1
        print(addr)

        receive_thread = threading.Thread(target=handle_receive, args = (client_socket, addr, user))
        receive_thread.daemon = True
        receive_thread.start()

def ExitKey():
    global loopVal
    loopVal = False

def InStreamToByteArray():
    print()

def genImage(promp, user):
    gene = gen4.genMain(promp, user)
    args = gene.args

    i = 0 # Iteration counter 반복 카운터
    j = 0 # Zoom video frame counter 줌 비디오 프레임 카운터
    p = 1 # Phrase counter 프레이즈 카운터
    this_video_frame = 0 # for video styling 영상 스타일링을 위해

    # Messing with learning rate / optimisers 학습률/옵티마이저 사용
    #variable_lr = args.step_size
    #optimiser_list = [['Adam',0.075],['AdamW',0.125],['Adagrad',0.2],['Adamax',0.125],['DiffGrad',0.075],['RAdam',0.125],['RMSprop',0.02]]

    # Do it
    try:
        with tqdm() as pbar:
            while True:            
                # Change generated image 생성된 이미지 변경
                if args.make_zoom_video:
                    if i % args.zoom_frequency == 0:
                        out = gene.synth(z)
                        
                        # Save image 이미지를 저장
                        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
                        img = np.transpose(img, (1, 2, 0))
                        imageio.imwrite('./steps/' + str(j) + '.png', np.array(img))

                        # Time to start zooming?   확대/축소를 시작할 시간입니까?
                        if args.zoom_start <= i:
                            # Convert z back into a Pil image  z를 다시 Pil 이미지로 변환
                            #pil_image = TF.to_pil_image(out[0].cpu())
                            
                            # Convert NP to Pil image NP를 필 이미지로 변환
                            pil_image = Image.fromarray(np.array(img).astype('uint8'), 'RGB')
                                                    
                            # Zoom
                            if args.zoom_scale != 1:
                                pil_image_zoom = gene.zoom_at(pil_image, gene.sideX/2, gene.sideY/2, args.zoom_scale)
                            else:
                                pil_image_zoom = pil_image
                            
                            # Shift - https://pillow.readthedocs.io/en/latest/reference/ImageChops.html
                            if args.zoom_shift_x or args.zoom_shift_y:
                                # This one wraps the image 이것은 이미지를 포장합니다.
                                pil_image_zoom = ImageChops.offset(pil_image_zoom, args.zoom_shift_x, args.zoom_shift_y)
                            
                            # Convert image back to a tensor again 이미지를 다시 텐서로 변환
                            pil_tensor = TF.to_tensor(pil_image_zoom)
                            
                            # Re-encode 다시 인코딩
                            z, *_ = gene.model.encode(pil_tensor.to(gene.device).unsqueeze(0) * 2 - 1)
                            z_orig = z.clone()
                            z.requires_grad_(True)

                            # Re-create optimiser 옵티마이저 재생성
                            opt = gene.get_opt(args.optimiser, args.step_size)
                        
                        # Next
                        j += 1
                
                # Change text prompt 텍스트 프롬프트 변경
                if args.prompt_frequency > 0:
                    if i % args.prompt_frequency == 0 and i > 0:
                        # In case there aren't enough phrases, just loop 구문이 충분하지 않은 경우 루프
                        if p >= len(gene.all_phrases):
                            p = 0
                        
                        pMs = []
                        args.prompts = gene.all_phrases[p]

                        # Show user we're changing prompt  사용자에게 변경 중인 프롬프트 표시
                        print(args.prompts)
                        
                        for prompt in args.prompts:
                            txt, weight, stop = gene.split_prompt(prompt)
                            embed = gene.perceptor.encode_text(gene.clip.tokenize(txt).to(gene.device)).float()
                            pMs.append(gene.Prompt(embed, weight, stop).to(gene.device))
                                            
                        p += 1

                # Training time 훈련 시간
                gene.train(i)
                
                # Ready to stop yet? 아직 멈출 준비가 되었나요?
                if i == args.max_iterations:
                    if not args.video_style_dir:
                        # we're done
                        break
                    else:                    
                        if this_video_frame == (gene.num_video_frames - 1):
                            # we're done
                            make_styled_video = True
                            break
                        else:
                            # Next video frame
                            this_video_frame += 1

                            # Reset the iteration count
                            i = -1
                            pbar.reset()
                                                    
                            # Load the next frame, reset a few options - same filename, different directory
                            args.init_image = gene.video_frame_list[this_video_frame]
                            print("Next frame: ", args.init_image)

                            if args.seed is None:
                                seed = torch.seed()
                            else:
                                seed = args.seed  
                            torch.manual_seed(seed)
                            print("Seed: ", seed)

                            filename = os.path.basename(args.init_image)
                            args.output = os.path.join(gene.cwd, "steps", filename)

                            # Load and resize image
                            img = Image.open(args.init_image)
                            pil_image = img.convert('RGB')
                            pil_image = pil_image.resize((gene.sideX, gene.sideY), Image.LANCZOS)
                            pil_tensor = TF.to_tensor(pil_image)
                            
                            # Re-encode
                            z, *_ = gene.model.encode(pil_tensor.to(gene.device).unsqueeze(0) * 2 - 1)
                            z_orig = z.clone()
                            z.requires_grad_(True)

                            # Re-create optimiser
                            opt = gene.get_opt(args.optimiser, args.step_size)

                i += 1
                pbar.update()
    except KeyboardInterrupt:
        pass

    # All done :)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = '\n genImageServer')
    parser.add_argument("-p", "--port", help = "port", default=22334)
    serArgs = parser.parse_args()
    try:
        port = serArgs.p
    except:
        pass

    try:
        thrGenImage()
    except KeyboardInterrupt:
        loopVal = False


