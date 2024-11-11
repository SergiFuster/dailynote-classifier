from setfit import SetFitModel

model = SetFitModel.from_pretrained("setfit-bge-small-v1.5-sst2-16-shot")

preds = model.predict([
    "Recordar llamar a la abuela el domingo por la tarde.",
    "Recordar pagar la factura del teléfono antes del 15 de noviembre.",
    "Reunión con el equipo de diseño el lunes a las 10:00 a.m.",
    "Cita con el dentista el próximo viernes a las 4:00 p.m.",
    "Comprar frutas y verduras frescas en el mercado el sábado por la mañana.",
    "Adquirir una tarjeta de regalo para el cumpleaños de Mariana.",
    "Enviar el informe mensual de progreso a la dirección antes del viernes.",
    "Preparar la presentación para el taller de inteligencia artificial la próxima semana.",
    "Revisar los apuntes de álgebra para el examen del martes.",
    "Completar el proyecto de programación antes del plazo final el jueves."
])

print(preds)