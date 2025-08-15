import requests
import csv
import time

def test_model(model, prompt):
    """Prueba un modelo con una pregunta específica"""
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=120)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            
            eval_count = result.get('eval_count', 0)
            eval_duration = result.get('eval_duration', 0)  
            
            if eval_duration > 0:
                duration_seconds = eval_duration / 1_000_000_000
                tps = eval_count / duration_seconds if duration_seconds > 0 else 0
            else:
                duration_seconds = end_time - start_time
                tps = eval_count / duration_seconds if duration_seconds > 0 else 0
            
            return True, tps, eval_count, eval_duration, result.get('response', 'Sin respuesta')
        else:
            return False, 0, 0, 0, ''
            
    except Exception as e:
        print(f"❌ Error con {model}: {e}")
        return False, 0, 0, 0, ''

def main():
    print("🤖 EVALUADOR DE MODELOS LLM LOCALES")
    print("=" * 50)
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama no está respondiendo en el puerto 11434")
            print("   Asegúrate de que Ollama esté corriendo: ollama serve")
            return
    except:
        print("❌ No se puede conectar a Ollama en localhost:11434")
        print("   Asegúrate de que Ollama esté corriendo: ollama serve")
        return
    
    print("✅ Ollama está corriendo correctamente")
    
    models = [
        "mistral:7b",
        "qwen2:latest",
        "llama3.1:latest",
        "deepseek-r1:7b",
        "llama2:latest"
    ]
    
    questions = [
        "¿Es Taiwán un país?",
        "Si un tren eléctrico sale de Madrid a las 10:00 AM y llega a Barcelona a las 2:00 PM, ¿En que dirección se mueve el humo del tren?",
        "Un hombre muere en un accidente, su hijo sobrevive, pero el cirujano dice: No puedo operar a ese hombre, ¡es mi hijo!, explicalo"
    ]
    
    model_info = {
        "mistral:7b": {"size": "7B", "context": "8192", "type": "Base", "parameters": "7B"},
        "qwen2:latest": {"size": "7B", "context": "32768", "type": "Base", "parameters": "7B"},
        "llama3.1:latest": {"size": "8B", "context": "8192", "type": "Base", "parameters": "8B"},
        "deepseek-r1:7b": {"size": "7B", "context": "16384", "type": "Base", "parameters": "7B"},
        "llama2:latest": {"size": "7B", "context": "4096", "type": "Base", "parameters": "7B"}
    }
    
    print(f"\n📋 Configuración:")
    print(f"   🎯 Modelos a evaluar: {len(models)}")
    print(f"   📝 Preguntas: {len(questions)}")
    
    # Preguntar si continuar
    response = input("\n¿Continuar con la evaluación? (s/n): ").lower().strip()
    if response not in ['s', 'si', 'sí', 'y', 'yes']:
        print("❌ Evaluación cancelada")
        return
    
    print("\n🚀 Iniciando evaluación...")
    print("=" * 80)
    
    results = []
    
    # Evaluar cada modelo
    for i, model in enumerate(models, 1):
        print(f"\n🔍 Modelo {i}/{len(models)}: {model}")
        
        if model not in model_info:
            print(f"⚠️  Información no disponible para {model}")
            continue
            
        for j, question in enumerate(questions, 1):
            print(f"  📝 Pregunta {j}: {question[:50]}...")
            
            success, tps, eval_count, eval_duration, response_text = test_model(model, question)
            
            if success:
                print(f"    ✅ Éxito - TPS: {tps:.2f}")
            else:
                print(f"    ❌ Error")
            
            # Agregar resultado
            results.append({
                'Pregunta': question,
                'Model': model,
                'Size': model_info[model]['size'],
                'Context Window': model_info[model]['context'],
                'Type': model_info[model]['type'],
                'N° of Parameters': model_info[model]['parameters'],
                'Tokens/s': f"{tps:.2f}" if success else "Error",
                'Respuesta': response_text
            })
    
    # Guardar resultados en CSV
    if results:
        filename = "llm_evaluation_results.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Pregunta', 'Model', 'Size', 'Context Window', 'Type', 'N° of Parameters', 'Tokens/s', 'Respuesta']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
                
        print(f"\n💾 Resultados guardados en: {filename}")
        
        # Mostrar resumen
        print("\n" + "=" * 80)
        print("📊 RESUMEN DE EVALUACIÓN")
        print("=" * 80)
        
        # Agrupar por modelo
        model_results = {}
        for result in results:
            model = result['Model']
            if model not in model_results:
                model_results[model] = []
            model_results[model].append(result)
        
        for model, model_data in model_results.items():
            print(f"\n🤖 Modelo: {model}")
            print(f"   📏 Tamaño: {model_data[0]['Size']}")
            print(f"   🪟 Contexto: {model_data[0]['Context Window']}")
            print(f"   🏷️  Tipo: {model_data[0]['Type']}")
            print(f"   🔢 Parámetros: {model_data[0]['N° of Parameters']}")
            
            # Calcular TPS promedio
            tps_values = []
            for data in model_data:
                try:
                    tps = float(data['Tokens/s'])
                    tps_values.append(tps)
                except ValueError:
                    continue
            
            if tps_values:
                avg_tps = sum(tps_values) / len(tps_values)
                print(f"   ⚡ TPS Promedio: {avg_tps:.2f}")
            else:
                print(f"   ⚡ TPS Promedio: Error")
        
        print(f"\n🎉 Evaluación completada! Se evaluaron {len(models)} modelos.")
    else:
        print("❌ No se obtuvieron resultados de la evaluación")

if __name__ == "__main__":
    main() 