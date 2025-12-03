
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import base64
import os
import threading
import queue
import requests
import concurrent.futures
from io import BytesIO
from PIL import Image
import speech_recognition as sr
import re

from openai import OpenAI
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'gior-secret-key-2025'
socketio = SocketIO(app, cors_allowed_origins="*")

class ChatSession:
    """Gerencia o histórico de conversa e contexto acumulado"""
    def __init__(self):
        self.historico = []
        self.imagens_enviadas = []
    
    def adicionar_mensagem(self, role, content, image_base64=None):
        mensagem = {"role": role, "content": content}
        if image_base64:
            self.imagens_enviadas.append(image_base64)
        self.historico.append(mensagem)
    
    def get_historico_formatado(self, consultor_contexto):
        """Retorna histórico formatado para a API da OpenAI"""
        messages = [{"role": "system", "content": consultor_contexto}]
        messages.extend(self.historico)
        return messages
    
    def limpar(self):
        self.historico = []
        self.imagens_enviadas = []


class GiorWeb:
    def __init__(self):
        self.ai_name = 'StyleVision - Consultoria de Moda'
        
        # Definição dos 3 consultores com personalidades distintas
        self.consultores = {
            "classico": {
                "nome": "O Clássico",
                "voz_tts": "alloy",
                "contexto": """Você é 'O Clássico', um consultor de moda tradicional e conservador, especializado em elegância atemporal.

Sua abordagem:
- Valoriza peças clássicas e atemporais
- Prioriza regras formais de etiqueta e dress codes
- Foca em elegância, sofisticação e bom gosto tradicional
- Tom respeitoso, educado e profissional

FORMATO OBRIGATÓRIO DE RESPOSTA (use exatamente estes 3 tópicos):

**Veredito:**
[Sua declaração sobre o look]

**O Problema:**
[O que precisa ser corrigido - se não houver problemas, diga "Nenhum problema identificado"]

**A Solução:**
[Sugestão prática para melhorar ou manter o look]

Fale em Português Brasileiro."""
            },
            "vanguardista": {
                "nome": "O Vanguardista", 
                "voz_tts": "nova",
                "contexto": """Você é 'O Vanguardista', um consultor de moda moderno, ousado e criativo.

Sua abordagem:
- Valoriza inovação, criatividade e experimentação
- Encoraja combinações inusitadas e tendências atuais
- Aprecia ousadia e expressão pessoal através da moda
- Tom inspirador, encorajador e energético

FORMATO OBRIGATÓRIO DE RESPOSTA (use exatamente estes 3 tópicos):

**Veredito:**
[Sua declaração sobre o look]

**O Problema:**
[O que precisa ser corrigido - se não houver problemas, diga "Nenhum problema identificado"]

**A Solução:**
[Sugestão prática para melhorar ou manter o look]

Fale em Português Brasileiro."""
            },
            "pragmatico": {
                "nome": "O Pragmático",
                "voz_tts": "onyx",
                "contexto": """Você é 'O Pragmático', um consultor de moda direto, funcional e brutalmente honesto.

Sua abordagem:
- Análise objetiva sem rodeios ou falsas cortesias
- Foca em praticidade, adequação e funcionalidade
- Críticas honestas sempre acompanhadas de soluções
- Tom direto, construtivo e sem floreios

FORMATO OBRIGATÓRIO DE RESPOSTA (use exatamente estes 3 tópicos):

**Veredito:**
[Sua declaração sobre o look]

**O Problema:**
[O que precisa ser corrigido - se não houver problemas, diga "Nenhum problema identificado"]

**A Solução:**
[Sugestão prática para melhorar ou manter o look]

Fale em Português Brasileiro."""
            }
        }
        
        self.sistema_ativo = False
        self.gravando_audio = False
        self.pergunta = ''
        self.imagem_capturada = None
        self.camera = None
        self.current_frame = None
        self.audio_queue = queue.Queue()
        
        # Chat sessions para cada consultor
        self.chat_sessions = {
            "classico": ChatSession(),
            "vanguardista": ChatSession(),
            "pragmatico": ChatSession()
        }
        
        # OpenAI client
        self.client = OpenAI()
        
        # Whisper - usando modelo menor e mais rápido
        print("Carregando modelo Whisper...")
        try:
            self.whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
            print("Modelo Whisper carregado!")
        except Exception as e:
            print(f"Aviso: Não foi possível carregar Whisper: {e}")
            self.whisper_model = None
        
        # Speech Recognition
        self.recognizer = sr.Recognizer()
        
        # Diretórios
        if not os.path.exists('frames'):
            os.makedirs('frames')
        if not os.path.exists('static'):
            os.makedirs('static')
        
        print(f"Sistema iniciado com 3 consultores: Clássico, Vanguardista e Pragmático")

    def ativar_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                self.camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            self.sistema_ativo = True
            return True
        return False

    def desativar_camera(self):
        if self.camera:
            self.camera.release()
            self.camera = None
            self.sistema_ativo = False
            return True
        return False

    def get_frame(self):
        if self.camera and self.sistema_ativo:
            success, frame = self.camera.read()
            if success:
                self.current_frame = frame.copy()
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                return frame_bytes
        return None

    def capturar_imagem(self):
        if self.current_frame is not None:
            cv2.imwrite('frames/captured_frame.jpg', self.current_frame)
            self.imagem_capturada = 'frames/captured_frame.jpg'
            
            # Salvar também na pasta static para exibição
            cv2.imwrite('static/captured.jpg', self.current_frame)
            return True
        return False

    def encode_image(self):
        try:
            image_path = self.imagem_capturada if self.imagem_capturada else 'frames/last_frame.jpg'
            
            if not os.path.exists(image_path):
                return None
            
            pil_image = Image.open(image_path).convert('RGB')
            max_side = 768
            w, h = pil_image.size
            if max(w, h) > max_side:
                scale = max_side / float(max(w, h))
                pil_image = pil_image.resize((int(w*scale), int(h*scale)))
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG", quality=80, optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # RETORNA A STRING BASE64 PURA, sem prefixo URL
            return img_str 
            
        except Exception as e:
            print(f"Erro ao codificar imagem: {e}")
            return None

    def formatar_mensagem_consultor(self, consultor_tipo, pergunta, encoded_image=None):
        """Formata mensagem com histórico de conversa para um consultor específico"""
        try:
            chat_session = self.chat_sessions[consultor_tipo]
            consultor_contexto = self.consultores[consultor_tipo]["contexto"]
            
            # Monta mensagens com histórico
            messages = chat_session.get_historico_formatado(consultor_contexto)
            
            # Adiciona pergunta atual
            if encoded_image:
                user_content = [
                    {"type": "text", "text": pergunta},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                    }
                ]
            else:
                user_content = pergunta
            
            messages.append({"role": "user", "content": user_content})
            
            return messages
            
        except Exception as e:
            print(f"Erro ao formatar mensagem para {consultor_tipo}: {e}")
            return None

    def obter_resposta_consultor(self, consultor_tipo, pergunta, encoded_image=None):
        """Obtém resposta de um consultor específico usando a API da OpenAI"""
        try:
            messages = self.formatar_mensagem_consultor(consultor_tipo, pergunta, encoded_image)
            
            if messages is None:
                return f"Erro ao formatar mensagem para {self.consultores[consultor_tipo]['nome']}"
            
            # Chama API da OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=300,
                temperature=0.7
            )
            
            resposta_texto = response.choices[0].message.content.strip()
            
            # Adiciona ao histórico da sessão
            self.chat_sessions[consultor_tipo].adicionar_mensagem(
                "user", 
                pergunta, 
                image_base64=encoded_image
            )
            self.chat_sessions[consultor_tipo].adicionar_mensagem(
                "assistant",
                resposta_texto
            )
            
            return resposta_texto
            
        except Exception as e:
            print(f"Erro ao obter resposta de {consultor_tipo}: {e}")
            return f"Erro ao consultar {self.consultores[consultor_tipo]['nome']}: {str(e)}"

    def gerar_audio_consultor(self, texto, consultor_tipo):
        """Gera áudio para a resposta de um consultor específico"""
        try:
            voz = self.consultores[consultor_tipo]["voz_tts"]
            
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=voz,
                input=texto,
                speed=1.3
            )
            
            audio_path = f'static/resposta_{consultor_tipo}.mp3'
            response.stream_to_file(audio_path)
            return audio_path
            
        except Exception as e:
            print(f"Erro ao gerar áudio para {consultor_tipo}: {e}")
            return None
    
    def limpar_historico(self, consultor_tipo=None):
        """Limpa histórico de conversa de um consultor específico ou de todos"""
        if consultor_tipo:
            self.chat_sessions[consultor_tipo].limpar()
        else:
            for tipo in self.consultores.keys():
                self.chat_sessions[tipo].limpar()

    def transcribe_audio(self, audio_file_path):
        try:
            if self.whisper_model is None:
                # Fallback para SpeechRecognition
                with sr.AudioFile(audio_file_path) as source:
                    audio = self.recognizer.record(source)
                    return self.recognizer.recognize_google(audio, language='pt-BR')
            
            segments, _ = self.whisper_model.transcribe(audio=audio_file_path, language='pt', beam_size=5)
            transcricao = ""
            for segment in segments:
                transcricao += segment.text + " "
            return transcricao.strip()
        except Exception as e:
            print(f"Erro na transcrição: {e}")
            return ""


# Instância global
gior = GiorWeb()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = gior.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('ativar_sistema')
def handle_ativar():
    success = gior.ativar_camera()
    emit('sistema_status', {'ativo': gior.sistema_ativo, 'mensagem': 'Sistema ativado!' if success else 'Câmera já está ativa'})

@socketio.on('desativar_sistema')
def handle_desativar():
    success = gior.desativar_camera()
    emit('sistema_status', {'ativo': gior.sistema_ativo, 'mensagem': 'Sistema desativado!' if success else 'Sistema já está inativo'})

@socketio.on('capturar_imagem')
def handle_capturar():
    if gior.capturar_imagem():
        emit('imagem_capturada', {'success': True, 'mensagem': 'Imagem capturada!', 'image_url': '/static/captured.jpg'})
    else:
        emit('imagem_capturada', {'success': False, 'mensagem': 'Erro ao capturar imagem'})

@socketio.on('processar_audio')
def handle_audio(data):
    try:
        # Receber áudio em base64
        audio_data = data['audio']

        # Decodificar e salvar
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        audio_path = 'static/temp_audio.wav'
        with open(audio_path, 'wb') as f:
            f.write(audio_bytes)

        # Transcrever
        transcricao = gior.transcribe_audio(audio_path)
        gior.pergunta = transcricao

        emit('transcricao_completa', {'transcricao': transcricao})

    except Exception as e:
        emit('erro', {'mensagem': f'Erro ao processar áudio: {str(e)}'})

@socketio.on('obter_descricao')
def handle_descricao():
    """Handler atualizado para trabalhar com 3 consultores"""
    try:
        pergunta = gior.pergunta if gior.pergunta else 'Como está meu look?'
        encoded_image = gior.encode_image()
        
        if not encoded_image:
            emit('erro', {'mensagem': 'Erro: Nenhuma imagem capturada. Por favor, capture ou envie uma foto.'})
            return
        
        print(f"Processando consulta com os 3 consultores: {pergunta}")
        emit('processando', {'mensagem': 'Consultando os 3 especialistas em moda...'})
        
        # Executar consultas aos 3 consultores em paralelo
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for tipo in ['classico', 'vanguardista', 'pragmatico']:
                futures[tipo] = executor.submit(
                    gior.obter_resposta_consultor, 
                    tipo, 
                    pergunta, 
                    encoded_image
                )
            
            # Coletar respostas
            respostas = {}
            for tipo, future in futures.items():
                try:
                    respostas[tipo] = future.result(timeout=30)
                except Exception as e:
                    print(f"Erro ao obter resposta de {tipo}: {e}")
                    respostas[tipo] = f"Erro ao consultar {gior.consultores[tipo]['nome']}"
        
        # Formatar HTML com as 3 análises
        def format_to_html(text):
            text = text.replace('\n', '<br>')
            text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
            text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
            return text
        
        feedback_html = ""
        for tipo in ['classico', 'vanguardista', 'pragmatico']:
            info = gior.consultores[tipo]
            resposta = respostas.get(tipo, "Erro ao obter resposta")
            
            feedback_html += f"""
                <div class="feedback-container consultor-{tipo}">
                    <h3>{info['nome']}</h3>
                    <p>{format_to_html(resposta)}</p>
                </div>
            """
        
        # Gerar áudios para os 3 consultores em paralelo
        audio_urls = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            audio_futures = {}
            for tipo in ['classico', 'vanguardista', 'pragmatico']:
                if respostas.get(tipo) and not respostas[tipo].startswith("Erro"):
                    audio_futures[tipo] = executor.submit(
                        gior.gerar_audio_consultor,
                        respostas[tipo],
                        tipo
                    )
            
            for tipo, future in audio_futures.items():
                try:
                    audio_path = future.result(timeout=15)
                    if audio_path:
                        audio_urls[tipo] = '/' + audio_path
                except Exception as e:
                    print(f"Erro ao gerar áudio para {tipo}: {e}")
        
        gior.pergunta = ''
        
        emit('descricao_completa', {
            'resposta': feedback_html,
            'audio_urls': audio_urls,
            'consultores': ['classico', 'vanguardista', 'pragmatico']
        })
        
    except Exception as e:
        print(f"Erro no handle_descricao: {e}")
        emit('erro', {'mensagem': f'Erro: {str(e)}'})

@socketio.on('limpar_historico')
def handle_limpar_historico(data=None):
    """Limpa histórico de um consultor específico ou de todos"""
    try:
        consultor_tipo = data.get('consultor') if data else None
        gior.limpar_historico(consultor_tipo)
        emit('historico_limpo', {
            'mensagem': f'Histórico limpo: {consultor_tipo if consultor_tipo else "todos"}' 
        })
    except Exception as e:
        emit('erro', {'mensagem': f'Erro ao limpar histórico: {str(e)}'})

@socketio.on('limpar_pergunta')
def handle_limpar():
    gior.pergunta = ''
    emit('pergunta_limpa', {'mensagem': 'Pergunta limpa'})

@socketio.on('upload_imagem')
def handle_upload(data):
    try:
        # Receber imagem em base64
        image_data = data['image']
        
        # Decodificar e salvar
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image_path = 'frames/uploaded_frame.jpg'
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        
        # Salvar também para exibição
        with open('static/captured.jpg', 'wb') as f:
            f.write(image_bytes)
        
        # Definir como imagem capturada
        gior.imagem_capturada = image_path
        
        emit('imagem_uploaded', {
            'success': True, 
            'mensagem': 'Imagem enviada com sucesso!',
            'image_url': '/static/captured.jpg'
        })
        
    except Exception as e:
        emit('erro', {'mensagem': f'Erro ao enviar imagem: {str(e)}'})

@app.route('/api/consultores', methods=['GET'])
def api_consultores():
    """Retorna informações sobre os 3 consultores disponíveis"""
    consultores_info = {}
    for tipo, info in gior.consultores.items():
        consultores_info[tipo] = {
            'nome': info['nome']
        }
    return jsonify({'consultores': consultores_info})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
